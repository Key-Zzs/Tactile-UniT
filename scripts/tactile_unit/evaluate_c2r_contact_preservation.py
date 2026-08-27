#!/usr/bin/env python3
"""Locked re-evaluation after the post-C2 Contact remediation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c2r_contact_preservation import (  # noqa: E402
    CONTACT_TRAINABLE_PREFIXES,
    canonical_contact_probe,
    retention,
    sha256_file,
    verify_accepted_c2_checkpoint,
)
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    different_episode_permutation,
    geometry_diagnostics,
    load_checkpoint,
    pairwise_alignment_metrics,
)
from gr00t.tactile_unit.trex_action_bootstrap import ReleasedTokenizerSource  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_continuous_vac_shared_space import (  # noqa: E402
    independent_audit,
    row_cosine,
    same_episode_wrong_time,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c2r_contact_preservation_remediation.json"
DEFAULT_C1 = ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--c1-config", type=Path, default=DEFAULT_C1)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--action-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt")
    parser.add_argument("--s2-checkpoint", type=Path, default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt")
    parser.add_argument("--s1-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt")
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def encode(model, split, device, batch_size):
    shared = {
        modality: np.empty((len(split), 8, 32), dtype=np.float32)
        for modality in ("vision", "action", "contact")
    }
    recovered_contact = np.empty_like(shared["contact"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            for modality, source in (("vision", "z_v"), ("action", "z_a"), ("contact", "z_c")):
                native = torch.from_numpy(np.array(split.arrays[source][start:stop], copy=True)).to(device)
                value = model.encode(modality, native)
                shared[modality][start:stop] = value.float().cpu().numpy()
                if modality == "contact":
                    recovered_contact[start:stop] = model.recover(modality, value).float().cpu().numpy()
    return shared, recovered_contact


def physics(decoder, split, recovered, device, batch_size):
    errors = np.empty(len(split), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            z_value = torch.from_numpy(recovered[start:stop]).to(device)
            current = torch.from_numpy(np.array(split.arrays["h_current"][start:stop], copy=True)).to(device)
            target = torch.from_numpy(np.array(split.arrays["h_future"][start:stop], copy=True)).to(device)
            prediction = decoder(z_value, current)
            errors[start:stop] = torch.square(prediction - target).mean(1).double().cpu().numpy()
    dynamic = np.asarray(split.arrays["dynamic"], dtype=bool)
    return {
        "future_mse": float(errors.mean()),
        "dynamic_mse": float(errors[dynamic].mean()),
    }


def subset_alignment(left, right, episode, mask, *, samples, seed, chunk):
    indices = np.flatnonzero(mask)
    if len(indices) < 2 or len(np.unique(episode[indices])) < 2:
        return {"count": len(indices), "status": "INSUFFICIENT"}
    return {
        "count": len(indices),
        **pairwise_alignment_metrics(
            left[indices], right[indices], episode[indices],
            bootstrap_samples=samples, seed=seed, retrieval_chunk=chunk,
        ),
    }


def state_boundary(baseline, selected):
    baseline_state = baseline.state_dict()
    selected_state = selected.state_dict()
    changed = []
    unchanged = []
    invalid = []
    for name, value in baseline_state.items():
        if torch.equal(value.cpu(), selected_state[name].cpu()):
            unchanged.append(name)
        else:
            changed.append(name)
            if not name.startswith(CONTACT_TRAINABLE_PREFIXES):
                invalid.append(name)
    return {"changed": changed, "unchanged": unchanged, "invalid_changed": invalid, "pass": not invalid}


def native_identity(args, c1, accepted_evaluation, s2, cache_root):
    expected = c1["frozen_identity"]
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    tokenizer = args.unit_checkpoint / "tokenizer"
    source = ReleasedTokenizerSource.open(tokenizer)
    actual = {
        "s1_teacher_file": sha256_file(args.s1_checkpoint),
        "s2_checkpoint_file": sha256_file(args.s2_checkpoint),
        "action_checkpoint_file": sha256_file(args.action_checkpoint),
        "old_action_rows": source.old_rows_digest(),
        "s2_encoder": parameter_digest(s2.encoder),
        "s2_decoder": parameter_digest(s2.decoder),
        "c1_cache_manifest": sha256_file(cache_root / "manifest.json"),
        **{
            f"vision_checkpoint/{name}": sha256_file(tokenizer / name)
            for name in expected["original_unit_tokenizer_files_sha256"]
        },
    }
    expected_values = {
        "s1_teacher_file": expected["s1_teacher_checkpoint_sha256"],
        "s2_checkpoint_file": expected["s2_checkpoint_sha256"],
        "action_checkpoint_file": expected["action_checkpoint_sha256"],
        "old_action_rows": expected["old_action_rows_digest"],
        "s2_encoder": expected["s2_encoder_parameter_digest"],
        "s2_decoder": expected["s2_decoder_parameter_digest"],
        "c1_cache_manifest": accepted_evaluation["c2r_metric_audit_cache_manifest"],
        **{
            f"vision_checkpoint/{name}": digest
            for name, digest in expected["original_unit_tokenizer_files_sha256"].items()
        },
    }
    equality = {name: actual[name] == digest for name, digest in expected_values.items()}
    return {"actual": actual, "expected": expected_values, "equality": equality, "pass": all(equality.values())}


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    c1 = json.loads(args.c1_config.read_text())
    runtime = config["runtime"]
    cache_root = ROOT / runtime["cache_root"]
    baseline_checkpoint = ROOT / runtime["accepted_c2_checkpoint"]
    selected_checkpoint = ROOT / runtime["experiment_root"] / "selected.pt"
    artifact_root = ROOT / runtime["artifact_root"]
    selection_path = artifact_root / "selection.json"
    selection_hash_file = artifact_root / "selection.sha256"
    if not selection_path.is_file() or not selection_hash_file.is_file():
        raise RuntimeError("C2-R selection was not frozen before test")
    expected_selection_hash = selection_hash_file.read_text().split()[0]
    actual_selection_hash = sha256_file(selection_path)
    if actual_selection_hash != expected_selection_hash:
        raise RuntimeError("frozen C2-R selection artifact hash mismatch")
    selection = json.loads(selection_path.read_text())
    if selection.get("test_loaded") is not False or selection.get("selection_split") != "validation only":
        raise RuntimeError("C2-R selection artifact permits test leakage")
    if sha256_file(selected_checkpoint) != selection["checkpoint_sha256"]:
        raise RuntimeError("selected C2-R checkpoint hash mismatch")
    baseline_sha = verify_accepted_c2_checkpoint(baseline_checkpoint)
    accepted = json.loads((ROOT / runtime["accepted_c2_evaluation"]).read_text())
    audit = json.loads((artifact_root / "metric_audit.json").read_text())
    if audit.get("decision") != "C2R0_METRIC_AUDIT_PASS":
        raise RuntimeError("C2-R0 audit is not valid")
    accepted_for_identity = dict(accepted)
    accepted_for_identity["c2r_metric_audit_cache_manifest"] = audit["cache_manifest_sha256_after"]

    # The locked test is intentionally not loaded until every selection lock above passes.
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=("1", "2", "3")
    )
    try:
        set_seed(int(config["seed"]))
        train = load_split(cache_root, "train", verify_hashes=True)
        test = load_split(cache_root, "test", verify_hashes=True)
        if len(test) != 17504:
            raise RuntimeError("locked C2 test row count changed")
        baseline, _ = load_checkpoint(baseline_checkpoint, device)
        selected, metadata = load_checkpoint(selected_checkpoint, device)
        baseline.eval().requires_grad_(False).to(device)
        selected.eval().requires_grad_(False).to(device)
        boundary = state_boundary(baseline, selected)
        baseline_shared, _ = encode(baseline, test, device, args.batch_size)
        selected_shared, recovered = encode(selected, test, device, args.batch_size)
        output_identity = {
            "vision": bool(np.array_equal(baseline_shared["vision"], selected_shared["vision"])),
            "action": bool(np.array_equal(baseline_shared["action"], selected_shared["action"])),
        }
        if not all(output_identity.values()) or not boundary["pass"]:
            raise RuntimeError("frozen Vision/Action identity failed")
        selected_shared_train = np.empty((len(train), 8, 32), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(train), args.batch_size):
                stop = min(start + args.batch_size, len(train))
                native = torch.from_numpy(np.array(train.arrays["z_c"][start:stop], copy=True)).to(device)
                selected_shared_train[start:stop] = selected.encode("contact", native).float().cpu().numpy()
        probes = {}
        for name, key, classes in (
            ("contact_transition", "contact_transition", 4),
            ("force_trend", "force_trend_class", 3),
        ):
            native_probe = canonical_contact_probe(
                train.arrays["z_c"], test.arrays["z_c"], train.arrays[key], test.arrays[key], classes
            )
            shared_probe = canonical_contact_probe(
                selected_shared_train, selected_shared["contact"], train.arrays[key], test.arrays[key], classes
            )
            probes[name] = {
                "native": native_probe,
                "shared": shared_probe,
                "retention": retention(shared_probe, native_probe),
            }

        s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
        selected_physics = physics(s2.decoder, test, recovered, device, args.batch_size)
        native_physics = physics(s2.decoder, test, np.asarray(test.arrays["z_c"]), device, args.batch_size)
        episode = np.asarray(test.arrays["episode_id"])
        transition = np.asarray(test.arrays["contact_transition"])
        masks = {
            "all": np.ones(len(test), dtype=bool),
            "dynamic": np.asarray(test.arrays["dynamic"], dtype=bool),
            "rare_boundary": np.isin(transition, [1, 2]),
            "free_to_contact": transition == 1,
            "contact_to_free": transition == 2,
        }
        bootstrap_samples = args.bootstrap_samples or int(config["evaluation"]["bootstrap_samples"])
        alignment = {}
        for offset, (name, left) in enumerate((("V-C", "vision"), ("A-C", "action"))):
            alignment[name] = {
                subset: subset_alignment(
                    selected_shared[left], selected_shared["contact"], episode, mask,
                    samples=bootstrap_samples,
                    seed=int(config["seed"]) + offset * 100 + index,
                    chunk=int(config["evaluation"]["retrieval_chunk"]),
                )
                for index, (subset, mask) in enumerate(masks.items())
            }
        pair_gates = {}
        for name, subsets in alignment.items():
            value = subsets["all"]
            multipliers = [
                value["retrieval"][direction]["recall_at_10"]
                / value["retrieval"][direction]["chance"]["recall_at_10"]
                for direction in ("forward", "reverse")
            ]
            pair_gates[name] = {
                "positive_margin_ci": value["paired_minus_shuffled_margin"] > 0
                and value["margin_bootstrap_ci95"][0] > 0,
                "retrieval": min(multipliers) >= float(config["evaluation"]["retrieval_r10_chance_multiplier_min"]),
                "r10_chance_multiplier": multipliers,
            }
            pair_gates[name]["pass"] = bool(
                pair_gates[name]["positive_margin_ci"] and pair_gates[name]["retrieval"]
            )
        wrong_time = same_episode_wrong_time(episode, np.asarray(test.arrays["t"]))
        random_pair = np.random.default_rng(int(config["seed"])).permutation(len(test))
        different = different_episode_permutation(episode, int(config["seed"]) + 11)
        controls = {}
        for name, left in (("V-C", "vision"), ("A-C", "action")):
            controls[name] = {
                "different_episode_shuffled_cosine": float(row_cosine(selected_shared[left], selected_shared["contact"][different]).mean()),
                "same_episode_wrong_time_cosine": float(row_cosine(selected_shared[left], selected_shared["contact"][wrong_time]).mean()),
                "modality_mismatched_random_cosine": float(row_cosine(selected_shared[left], selected_shared["contact"][random_pair]).mean()),
            }
        reversed_code = np.empty_like(selected_shared["contact"])
        reversed_shared = np.empty_like(selected_shared["contact"])
        with torch.inference_mode():
            for start in range(0, len(test), args.batch_size):
                stop = min(start + args.batch_size, len(test))
                current = torch.from_numpy(np.array(test.arrays["h_current"][start:stop], copy=True)).to(device)
                future = torch.from_numpy(np.array(test.arrays["h_future"][start:stop], copy=True)).to(device)
                code = s2.encoder(future, current)
                reversed_code[start:stop] = code.float().cpu().numpy()
                reversed_shared[start:stop] = selected.encode("contact", code).float().cpu().numpy()
        controls["reversed_contact_transition"] = {
            "native_paired_cosine": float(row_cosine(test.arrays["z_c"], reversed_code).mean()),
            "shared_paired_cosine": float(row_cosine(selected_shared["contact"], reversed_shared).mean()),
        }
        contact_geometry = geometry_diagnostics(selected_shared["contact"])
        collapse = bool(
            contact_geometry["per_dimension_variance"]["near_zero_fraction"] >= 0.5
            or contact_geometry["query_diversity"]["collapsed_pair_fraction"] >= 0.5
        )
        identities = native_identity(args, c1, accepted_for_identity, s2, cache_root)
        independent = independent_audit(selected)
        contact_gate = probes["contact_transition"]["retention"] >= float(config["evaluation"]["contact_retention_min"])
        force_gate = probes["force_trend"]["retention"] >= float(config["evaluation"]["force_retention_min"])
        alignment_gate = all(value["pass"] for value in pair_gates.values())
        structural = bool(
            boundary["pass"] and all(output_identity.values()) and identities["pass"]
            and independent["status"] == "PASS" and not collapse
        )
        original_physics = accepted["contact"]["reconstruction"]["recovered"]
        physics_improved = bool(
            selected_physics["future_mse"] < original_physics["future_mse"]
            and selected_physics["dynamic_mse"] < original_physics["dynamic_mse"]
        )
        if not structural:
            decision = "STRUCTURAL_FAIL"
        elif not alignment_gate:
            decision = "C2R_ALIGNMENT_REGRESSION"
        elif contact_gate and force_gate:
            decision = (
                "C2R_SHARED_SPACE_READY" if physics_improved
                else "C2R_SHARED_SPACE_READY_WITH_PHYSICS_WARNING"
            )
        else:
            decision = "C2R_DUAL_PATH_RECOMMENDED"
        result = {
            "schema": "tactile3d-unit.vac-c2r-locked-evaluation.v1",
            "decision": decision,
            "evaluation_type": "LOCKED RE-EVALUATION AFTER POST-C2 REMEDIATION",
            "first_look_untouched_test": False,
            "rows": len(test),
            "selection_frozen_before_test": True,
            "selection_artifact_sha256": actual_selection_hash,
            "selected": {**selection, "checkpoint_metadata": metadata},
            "accepted_c2_checkpoint_sha256": baseline_sha,
            "gpu": gpu,
            "probes": probes,
            "physics": {
                "native": native_physics,
                "original_c2": original_physics,
                "c2r": selected_physics,
                "improved_over_c2": physics_improved,
            },
            "alignment": alignment,
            "accepted_v_a_alignment": accepted["alignment"]["V-A"],
            "pair_gates": pair_gates,
            "negative_controls": controls,
            "contact_geometry": contact_geometry,
            "state_boundary": boundary,
            "frozen_output_identity": output_identity,
            "action_integrity": {
                "identity": output_identity["action"],
                "accepted_metrics_unchanged": accepted["action"],
            },
            "vision_integrity": {
                "identity": output_identity["vision"],
                "accepted_metrics_unchanged": accepted["vision"],
            },
            "native_identities": identities,
            "independent_encodability": independent,
            "gates": {
                "contact": contact_gate,
                "force": force_gate,
                "alignment": alignment_gate,
                "noncollapse": not collapse,
                "structural": structural,
                "physics_improved": physics_improved,
            },
            "scope": {
                "C3": "NOT STARTED", "C4": "NOT STARTED", "C5": "NOT STARTED",
                "C6_M3": "NOT STARTED", "M3": "NOT ESTABLISHED",
            },
        }
        atomic_json(artifact_root / "locked_test_evaluation.json", result)
        atomic_json(artifact_root / "final_decision.json", {
            "schema": "tactile3d-unit.vac-c2r-decision.v1",
            "decision": decision,
            "reasons": {
                "contact_retention": probes["contact_transition"]["retention"],
                "force_retention": probes["force_trend"]["retention"],
                "alignment_gate": alignment_gate,
                "physics_improved": physics_improved,
                "structural": structural,
            },
            "C3": "NOT STARTED",
            "M3": "NOT ESTABLISHED",
        })
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
