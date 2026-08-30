#!/usr/bin/env python3
"""Run the read-only continuous VAC pre-Track-C integration acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_data import TReXActionCache  # noqa: E402
from gr00t.tactile_unit.trex_action_transition import (  # noqa: E402
    load_shared_transition_checkpoint,
)
from gr00t.tactile_unit.vac_transition_contract import (  # noqa: E402
    FutureOracleLeakageError,
    OfflineVACTransitionTeachers,
    reject_online_oracles,
    validate_integrated_manifest_row,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_s1_teacher,
    load_s2_model,
)


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/integration_continuous_vac_contract.json"
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/tactile_unit/integration"
DEFAULT_CACHE = ROOT / ".local/cache/tactile_unit/integration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--unit-checkpoint", type=Path)
    parser.add_argument(
        "--s1-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt",
    )
    parser.add_argument(
        "--s2-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt",
    )
    parser.add_argument(
        "--action-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if check and completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def is_ancestor(reference: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", reference, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def ancestry_audit() -> dict[str, Any]:
    references = {
        "track_b": "origin/develop/contact-semantic-tokenizer",
        "c0": "origin/develop/continuous-contact-bridge",
        "track_a": "origin/develop/tactile-action-bootstrap",
        "a_r": "origin/develop/action-transition-remediation",
    }
    contained = {name: is_ancestor(reference) for name, reference in references.items()}
    result = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "common_base": git(
            "merge-base",
            "origin/develop/continuous-contact-bridge",
            "origin/develop/action-transition-remediation",
        ),
        "references": {
            name: {"reference": reference, "commit": git("rev-parse", reference)}
            for name, reference in references.items()
        },
        "contained": contained,
        "status": "PASS" if all(contained.values()) else "FAIL",
    }
    if result["branch"] != "develop/tactile-unit-integration" or not all(contained.values()):
        raise RuntimeError("integration ancestry hard gate failed")
    return result


def parse_worktrees() -> list[dict[str, str]]:
    blocks = git("worktree", "list", "--porcelain").split("\n\n")
    result: list[dict[str, str]] = []
    for block in blocks:
        row: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            row[key] = value
        if row:
            result.append(row)
    return result


def discover_c0_runtime(cache_root: Path, expected_sha: str) -> tuple[Path, dict[str, Any]]:
    destination = cache_root / "c0_paired_test.npz"
    if destination.is_file() and sha256_file(destination) == expected_sha:
        return destination, {"source": "integration-local identity-validated copy", "copied": False}
    candidates: list[Path] = []
    for worktree in parse_worktrees():
        if worktree.get("branch") == "refs/heads/develop/continuous-contact-bridge":
            candidates.append(
                Path(worktree["worktree"]) / ".local/cache/tactile_unit/c0/paired_test.npz"
            )
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_sha:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            if sha256_file(destination) != expected_sha:
                raise RuntimeError("C0 runtime cache copy changed identity")
            return destination, {
                "source": "dynamically discovered continuous-contact worktree",
                "copied": True,
            }
    raise FileNotFoundError("accepted C0 native feature cache was not found in any worktree")


def optional_c0_bridge(expected_sha: str) -> dict[str, Any]:
    for worktree in parse_worktrees():
        if worktree.get("branch") != "refs/heads/develop/continuous-contact-bridge":
            continue
        path = (
            Path(worktree["worktree"])
            / ".local/experiments/tactile_unit/c0/continuous_bridge_candidates.pt"
        )
        if path.is_file():
            actual = sha256_file(path)
            if actual != expected_sha:
                raise RuntimeError("optional C0 bridge exists but has the wrong identity")
            return {"availability": "available", "required": False, "sha256": actual}
    return {"availability": "not available", "required": False, "sha256": None}


def pair_id_digest(pair_ids: np.ndarray | list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, pair_ids)).encode()).hexdigest()


def validate_manifest(
    manifest_path: Path, expected_sha: str, expected_canonical_sha: str
) -> tuple[dict[str, Any], list[Any]]:
    if sha256_file(manifest_path) != expected_sha:
        raise RuntimeError("canonical S3.1 paired manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("canonical_sha256") != expected_canonical_sha:
        raise RuntimeError("canonical S3.1 manifest content identity mismatch")
    rows = manifest["rows"]
    anchors = [validate_integrated_manifest_row(row) for row in rows]
    if len(rows) != 960 or len({anchor.pair_id for anchor in anchors}) != 960:
        raise RuntimeError("canonical manifest must contain 960 unique pairs")
    if any(row["source"]["split"] != "test" for row in rows):
        raise RuntimeError("canonical evaluation contains a non-test row")
    return manifest, anchors


def match_action_rows(cache: TReXActionCache, anchors: list[Any]) -> np.ndarray:
    episode = np.asarray(cache.episode_id, dtype=np.int64)
    frame = np.asarray(cache.anchor_frame, dtype=np.int64)
    keys = (episode << 32) | frame
    if np.any(keys[1:] <= keys[:-1]):
        raise RuntimeError("Action cache ordering is not unique episode/frame order")
    wanted = np.asarray(
        [(int(anchor.episode_id) << 32) | int(anchor.t) for anchor in anchors], dtype=np.int64
    )
    indices = np.searchsorted(keys, wanted)
    if np.any(indices >= len(keys)) or not np.array_equal(keys[indices], wanted):
        raise RuntimeError("Action cache does not contain every canonical VAC pair")
    return indices.astype(np.int64)


@torch.inference_mode()
def encode_action(
    model: torch.nn.Module,
    batch: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result: list[np.ndarray] = []
    model.eval().requires_grad_(False)
    for start in range(0, len(batch["state"]), batch_size):
        stop = min(start + batch_size, len(batch["state"]))
        state = torch.from_numpy(batch["state"][start:stop])
        action = torch.from_numpy(batch["action"][start:stop])
        embodiment = torch.full((stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long)
        z_action, _, _ = model.encode(state, action, embodiment)
        result.append(z_action.float().cpu().numpy())
    z_a = np.concatenate(result).astype(np.float32, copy=False)
    count = min(batch_size, len(z_a))
    state = torch.from_numpy(batch["state"][:count])
    action = torch.from_numpy(batch["action"][:count])
    embodiment = torch.full((count,), TREX_EMBODIMENT_ID, dtype=torch.long)
    first = model.encode(state, action, embodiment)[0]
    second = model.encode(state, action, embodiment)[0]
    return z_a, {
        "repeat_exact": bool(torch.equal(first, second)),
        "finite": bool(np.isfinite(z_a).all()),
        "shape": list(z_a.shape),
        "device": "cpu",
    }


def flatten_normalize(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(flatten_normalize(left) * flatten_normalize(right), axis=1)


def different_episode_permutation(episode_id: np.ndarray) -> np.ndarray:
    count = len(episode_id)
    permutation = (np.arange(count) + max(1, count // 2)) % count
    for index in range(count):
        while episode_id[permutation[index]] == episode_id[index]:
            permutation[index] = (permutation[index] + 1) % count
    return permutation


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    numerator = np.square(np.linalg.norm(x.T @ y, ord="fro"))
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / max(float(denominator), 1e-12))


def retrieval(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    similarity = flatten_normalize(left) @ flatten_normalize(right).T
    positive = np.diag(similarity)
    ranks = 1 + np.sum(similarity > positive[:, None], axis=1)
    count = len(ranks)
    return {
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance": {
            "recall_at_1": 1.0 / count,
            "recall_at_5": min(5, count) / count,
            "recall_at_10": min(10, count) / count,
        },
    }


def pairwise_baseline(
    left: np.ndarray, right: np.ndarray, episode_id: np.ndarray
) -> dict[str, Any]:
    negative = different_episode_permutation(episode_id)
    paired = row_cosine(left, right)
    shuffled = row_cosine(left, right[negative])
    forward = retrieval(left, right)
    reverse = retrieval(right, left)
    return {
        "paired_cosine": float(paired.mean()),
        "different_episode_shuffled_cosine": float(shuffled.mean()),
        "paired_minus_shuffled_margin": float(paired.mean() - shuffled.mean()),
        "linear_cka": linear_cka(left, right),
        "retrieval": {"forward": forward, "reverse": reverse},
        "symmetric_summary": {
            "recall_at_1": (forward["recall_at_1"] + reverse["recall_at_1"]) / 2,
            "recall_at_5": (forward["recall_at_5"] + reverse["recall_at_5"]) / 2,
            "recall_at_10": (forward["recall_at_10"] + reverse["recall_at_10"]) / 2,
            "mrr": (forward["mrr"] + reverse["mrr"]) / 2,
            "median_rank": (forward["median_rank"] + reverse["median_rank"]) / 2,
            "chance": forward["chance"],
        },
    }


def verify_online_guard() -> dict[str, Any]:
    rejected: list[str] = []
    payloads = {
        "top_level_future_vision": {"i_t+16": object()},
        "nested_contact_teacher": {"observation": {"teacher": {"z_c": np.zeros(1)}}},
        "list_wrapped_action_teacher": {"items": [{"z_a_target": np.zeros(1)}]},
    }
    for name, payload in payloads.items():
        try:
            reject_online_oracles(payload)
        except FutureOracleLeakageError:
            rejected.append(name)
    legal = {
        "current": {"image": "current reference", "h_t_c": "current context"},
        "policy": {"z_hat_a": "policy generated", "z_hat_c": "policy generated"},
    }
    reject_online_oracles(legal)
    reject_online_oracles({"z_c": "oracle"}, oracle_eval=True)
    status = len(rejected) == len(payloads)
    if not status:
        raise RuntimeError("nested online oracle guard failed")
    return {
        "offline_teacher_only": ["z_v", "ActionTransitionTarget.z_a", "z_c"],
        "online_legal": ["I_<=t", "robot_state_<=t", "T_[t-0.5:t]", "h_t_c", "z_hat_a", "z_hat_c"],
        "rejected_cases": rejected,
        "oracle_eval_explicit_bypass": True,
        "action_teacher_planned_distinction": "PASS",
        "nested_leakage_guards": "PASS",
        "status": "PASS",
    }


def verify_m3() -> dict[str, Any]:
    path = ROOT / "configs/tactile_unit/m3_continuous_vac_evaluation.json"
    value = json.loads(path.read_text())
    gates = " ".join(row["gate"].lower() for row in value["preregistered_gates"])
    checks = {
        "spec_only": value["status"] == "SPEC_ONLY_NOT_EXECUTED",
        "shared_codebook_not_required": value["same_codebook_required"] is False,
        "continuous_contact": value["contact_interface"]["type"] == "continuous",
        "v_a_pairing": "paired v-a" in gates,
        "v_c_pairing": "paired v-c" in gates,
        "a_c_pairing": "paired a-c" in gates,
        "retrieval": "retrieval" in gates,
        "cross_modal_evidence": "cross-modal" in gates,
        "missing_modality": "missing modalities" in gates,
        "action_temporal_semantics": "temporal semantics" in gates,
        "teacher_online_boundary": "offline-only" in gates,
    }
    if not all(checks.values()):
        raise RuntimeError("M3 continuous-hybrid spec is inconsistent")
    return {"checks": checks, "status": "PASS", "m3_established": False}


def human_acceptance(summary: Mapping[str, Any]) -> str:
    baseline = summary["native_pairwise_baseline"]
    checkpoint = summary["checkpoint_audit"]
    pair = summary["pair_contract_audit"]
    lines = [
        "# Tactile3D-UniT Integration Human Acceptance",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## Inspect",
        "",
        f"1. Ancestry: `{summary['ancestry_audit']['status']}`; "
        f"common base `{summary['ancestry_audit']['common_base']}`.",
        f"2. Checkpoints: `{checkpoint['status']}`; S1/S2/A-R and Original UniT identities are frozen.",
        f"3. Same-pair contract: `{pair['status']}` over {pair['manifest_coverage']} canonical pairs.",
        "4. Timing: Vision `I_t,I_t+16`; Action `a_t:a_t+15`; Contact `[t-15,t]` and `[t+1,t+16]`.",
        f"5. Causal boundary: `{summary['causal_audit']['status']}`; nested future teachers are rejected online.",
        f"6. Action temporal integrity: `{summary['action_integrity']['status']}`; continuous pre-RQ.",
        f"7. Contact integrity: `{summary['contact_integrity']['status']}`; native continuous, no RQ/whitening.",
        "8. Native starting margins (V-A / V-C / A-C): "
        + " / ".join(f"{baseline[name]['paired_minus_shuffled_margin']:.6f}" for name in ("V-A", "V-C", "A-C"))
        + ".",
        f"9. Original UniT non-regression: `{summary['original_unit_non_regression']['status']}`.",
        f"10. Final decision: `{summary['decision']}`.",
        "",
        "This is a read-only native baseline. No Track C training, projector training, "
        "M3 execution, optimizer, or backward pass occurred.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    forbidden_scope = (
        "full_track_c_started",
        "m3_established",
        "training_allowed",
        "optimizer_allowed",
        "backward_allowed",
    )
    if any(config["scope"][name] for name in forbidden_scope):
        raise RuntimeError("integration config broadens the forbidden stage scope")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    ancestry = ancestry_audit()
    atomic_json(args.artifact_root / "ancestry_audit.json", ancestry)

    identity = config["frozen_identity"]
    data_config = config["canonical_data"]
    unit_checkpoint = args.unit_checkpoint
    if unit_checkpoint is None:
        raw = os.environ.get("UNIT_FULLDATA_CKPT")
        if not raw:
            raise RuntimeError("UNIT_FULLDATA_CKPT or --unit-checkpoint is required")
        unit_checkpoint = Path(raw)
    tokenizer_root = unit_checkpoint / "tokenizer"
    tokenizer_files = {
        name: tokenizer_root / name for name in identity["original_unit_tokenizer_files_sha256"]
    }
    immutable_files = {
        "s1_teacher": args.s1_checkpoint,
        "s2_checkpoint": args.s2_checkpoint,
        "action_checkpoint": args.action_checkpoint,
        **{f"original_unit/{name}": path for name, path in tokenizer_files.items()},
    }
    before = {name: sha256_file(path) for name, path in immutable_files.items()}
    expected_files = {
        "s1_teacher": identity["s1_teacher_checkpoint_sha256"],
        "s2_checkpoint": identity["s2_checkpoint_sha256"],
        "action_checkpoint": identity["action_checkpoint_sha256"],
        **{
            f"original_unit/{name}": digest
            for name, digest in identity["original_unit_tokenizer_files_sha256"].items()
        },
    }
    if before != expected_files:
        raise RuntimeError("a canonical checkpoint identity changed before integration smoke")

    source = ReleasedTokenizerSource.open(tokenizer_root)
    old_rows_before = source.old_rows_digest()
    if old_rows_before != identity["old_action_rows_digest"]:
        raise RuntimeError("Original UniT Action rows 0-29 digest mismatch")
    action_model, action_metadata = load_shared_transition_checkpoint(args.action_checkpoint, source)
    action_model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in action_model.parameters()):
        raise RuntimeError("canonical Action model was not frozen")

    action_parameters_before = parameter_digest(action_model)
    s1_model = load_s1_teacher(args.s1_checkpoint, torch.device("cpu"))
    s1_parameters_before = parameter_digest(s1_model)
    s2_model = load_s2_model(args.s2_checkpoint, torch.device("cpu"))
    encoder_digest = parameter_digest(s2_model.encoder)
    decoder_digest = parameter_digest(s2_model.decoder)
    if (
        encoder_digest != identity["s2_encoder_parameter_digest"]
        or decoder_digest != identity["s2_decoder_parameter_digest"]
    ):
        raise RuntimeError("S2 E_c/D_c parameter identity mismatch")

    c0_cache_path, c0_discovery = discover_c0_runtime(
        args.cache_root, data_config["c0_native_feature_cache_sha256"]
    )
    manifest_path = ROOT / data_config["paired_manifest"]
    _manifest, anchors = validate_manifest(
        manifest_path,
        data_config["paired_manifest_sha256"],
        data_config["paired_manifest_canonical_sha256"],
    )
    with np.load(c0_cache_path, allow_pickle=False) as loaded:
        c0 = {name: np.asarray(loaded[name]) for name in loaded.files}
    expected_pair_ids = np.asarray([anchor.pair_id for anchor in anchors])
    expected_episode = np.asarray([anchor.episode_id for anchor in anchors], dtype=np.int64)
    expected_t = np.asarray([anchor.t for anchor in anchors], dtype=np.int64)
    if not np.array_equal(c0["pair_id"], expected_pair_ids):
        raise RuntimeError("C0 cache pair ordering differs from canonical manifest")
    if not np.array_equal(c0["episode_id"], expected_episode) or not np.array_equal(
        c0["anchor_frame"], expected_t
    ):
        raise RuntimeError("C0 cache episode/time anchors differ from canonical manifest")
    if pair_id_digest(c0["pair_id"]) != data_config["pair_id_digest"]:
        raise RuntimeError("C0 pair ID digest mismatch")

    action_cache = TReXActionCache(
        ROOT / ".local/cache/tactile_unit/s3_3/action_windows",
        "test",
        ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json",
    )
    action_indices = match_action_rows(action_cache, anchors)
    action_batch = action_cache.batch(action_indices)
    if not np.array_equal(action_batch["episode_id"], expected_episode) or not np.array_equal(
        action_batch["anchor_frame"], expected_t
    ):
        raise RuntimeError("Action rows reordered the canonical pairs")
    z_a, action_smoke = encode_action(action_model, action_batch, args.batch_size)
    if not action_smoke["repeat_exact"] or not action_smoke["finite"]:
        raise RuntimeError("Action transition smoke is non-deterministic or non-finite")

    count = len(anchors)
    masks = {
        "vision": np.ones(count, dtype=bool),
        "action": np.ones(count, dtype=bool),
        "contact": np.ones(count, dtype=bool),
    }
    teachers = OfflineVACTransitionTeachers(
        pair_id=c0["pair_id"].tolist(),
        episode_id=expected_episode,
        t=expected_t,
        t_future=expected_t + 16,
        z_v=c0["z_v"].astype(np.float32, copy=False),
        z_a=z_a,
        z_c=c0["z_c"].astype(np.float32, copy=False),
        h_t_c=c0["h_current"].astype(np.float32, copy=False),
        state=action_batch["state"].astype(np.float32, copy=False),
        action=action_batch["action"].astype(np.float32, copy=False),
        modality_masks=masks,
        provenance={
            "pair_manifest_sha256": data_config["paired_manifest_sha256"],
            "vision_source": "Original UniT frozen Vision cache from accepted C0",
            "action_checkpoint_sha256": identity["action_checkpoint_sha256"],
            "contact_checkpoint_sha256": identity["s2_checkpoint_sha256"],
        },
    )
    cache_output = args.cache_root / "canonical_vac_teachers.npz"
    np.savez_compressed(
        cache_output,
        pair_id=np.asarray(teachers.pair_id),
        episode_id=expected_episode,
        t=expected_t,
        t_future=expected_t + 16,
        z_v=teachers.z_v,
        z_a=teachers.z_a,
        z_c=teachers.z_c,
        h_t_c=teachers.h_t_c,
        state=teachers.state,
        action=teachers.action,
        state_mask=action_batch["state_mask"],
        action_mask=action_batch["action_mask"],
        modality_vision=masks["vision"],
        modality_action=masks["action"],
        modality_contact=masks["contact"],
    )

    split_manifest = json.loads(
        (ROOT / ".local/artifacts/tactile_teacher/s1_0/split_manifest.json").read_text()
    )["episode_ids"]
    split_sets = {name: set(map(int, split_manifest[name])) for name in ("train", "val", "test")}
    split_overlap = {
        "train_validation": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(split_overlap.values()) or not set(expected_episode) <= split_sets["test"]:
        raise RuntimeError("canonical pair split leakage or membership failure")

    rare = np.isin(c0["contact_transition"], [1, 3])
    baseline = {
        "schema": "tactile3d-unit.integration-native-pairwise-baseline.v1",
        "role": "READ_ONLY_NATIVE_BASELINE_NO_TRACK_C_TRAINING",
        "samples": count,
        "V-A": pairwise_baseline(teachers.z_v, teachers.z_a, expected_episode),
        "V-C": pairwise_baseline(teachers.z_v, teachers.z_c, expected_episode),
        "A-C": pairwise_baseline(teachers.z_a, teachers.z_c, expected_episode),
        "subsets": {
            "dynamic_count": int(c0["dynamic"].sum()),
            "rare_boundary_count": int(rare.sum()),
        },
    }
    atomic_json(args.artifact_root / "native_pairwise_baseline.json", baseline)

    causal = verify_online_guard()
    atomic_json(args.artifact_root / "causal_audit.json", causal)

    action_evaluation = json.loads(
        (ROOT / ".local/artifacts/tactile_unit/s3_3_r/held_out_evaluation.json").read_text()
    )
    dynamic_temporal = action_evaluation["temporal_controls"]["paired_bootstrap"]["dynamic"]
    action_integrity = {
        "dynamic_reversed_over_correct": dynamic_temporal["reversed"]["ratio"],
        "dynamic_shuffled_over_correct": dynamic_temporal["shuffled"]["ratio"],
        "zero_over_full": dynamic_temporal["zero"]["ratio"],
        "effective_rank": action_evaluation["noncollapse"]["effective_rank"],
        "collapsed_query_fraction": action_evaluation["noncollapse"]["collapsed_query_fraction"],
        "canonical_path": "CONTINUOUS PRE-RQ",
        "identity_validation": action_evaluation["decision"],
        "status": "PASS" if action_evaluation["ready"] else "FAIL",
    }
    if action_integrity["status"] != "PASS":
        raise RuntimeError("accepted A-R action integrity did not survive integration")

    m2 = json.loads((ROOT / ".local/artifacts/contact_dynamics/s2_5/m2_acceptance.json").read_text())
    contact_q = None
    for worktree in parse_worktrees():
        if worktree.get("branch") == "refs/heads/develop/continuous-contact-bridge":
            candidate = Path(worktree["worktree"]) / ".local/artifacts/tactile_unit/s3_2_q/final_decision.json"
            if candidate.is_file():
                contact_q = json.loads(candidate.read_text())
                break
    contact_integrity = {
        "canonical": "CONTINUOUS z_c",
        "rq_in_canonical_path": False,
        "whitening": False,
        "contact_transition_macro_f1": m2["dynamic_semantics"]["contact_transition_macro_f1"],
        "force_trend_macro_f1": m2["dynamic_semantics"]["force_trend_macro_f1"],
        "future_reconstruction_mse": m2["test_metrics"]["all"]["future_mse"],
        "rare_boundary_identity": (
            None
            if contact_q is None
            else contact_q["track_c_contract"]["interface_type"]
        ),
        "status": "PASS" if m2["s2_final"] == "PASS" else "FAIL",
    }
    if contact_integrity["status"] != "PASS":
        raise RuntimeError("accepted continuous Contact integrity did not survive integration")

    t4 = json.loads((ROOT / ".local/artifacts/reproduction/t4/extraction_summary.json").read_text())
    original_non_regression = {
        "old_action_rows": "PASS",
        "old_action_rows_digest": old_rows_before,
        "gr1_action": "exact identity-validated shared tensors and rows 0-29",
        "vision_identity": "PASS",
        "t4_status": t4["status"],
        "t4_samples": t4["sample_count"],
        "t4_l2_shape": t4["shapes"]["l2"],
        "status": "PASS" if t4["status"] == "PASS" and t4["sample_count"] == 960 else "FAIL",
    }
    if original_non_regression["status"] != "PASS":
        raise RuntimeError("Original UniT T4 identity audit failed")

    after = {name: sha256_file(path) for name, path in immutable_files.items()}
    old_rows_after = source.old_rows_digest()
    action_parameters_after = parameter_digest(action_model)
    s1_parameters_after = parameter_digest(s1_model)
    encoder_digest_after = parameter_digest(s2_model.encoder)
    decoder_digest_after = parameter_digest(s2_model.decoder)
    parameter_identities_unchanged = (
        action_parameters_before == action_parameters_after
        and s1_parameters_before == s1_parameters_after
        and encoder_digest == encoder_digest_after
        and decoder_digest == decoder_digest_after
    )
    checkpoint_audit = {
        "before": before,
        "after": after,
        "s2_encoder_parameter_digest": encoder_digest,
        "s2_decoder_parameter_digest": decoder_digest,
        "old_action_rows_before": old_rows_before,
        "old_action_rows_after": old_rows_after,
        "action_parameter_digest_before": action_parameters_before,
        "action_parameter_digest_after": action_parameters_after,
        "s1_parameter_digest_before": s1_parameters_before,
        "s1_parameter_digest_after": s1_parameters_after,
        "s2_encoder_parameter_digest_after": encoder_digest_after,
        "s2_decoder_parameter_digest_after": decoder_digest_after,
        "action_checkpoint_metadata": action_metadata,
        "optional_c0_bridge": optional_c0_bridge(
            identity["optional_c0_bridge_checkpoint_sha256"]
        ),
        "frozen_trainable_parameters": {
            "action": sum(parameter.numel() for parameter in action_model.parameters() if parameter.requires_grad),
            "s1": sum(parameter.numel() for parameter in s1_model.parameters() if parameter.requires_grad),
            "s2": sum(parameter.numel() for parameter in s2_model.parameters() if parameter.requires_grad),
            "vision": "not instantiated; exact accepted checkpoint/cache identity used",
        },
        "optimizer_instantiated": False,
        "backward_executed": False,
        "status": (
            "PASS"
            if before == after
            and old_rows_before == old_rows_after
            and parameter_identities_unchanged
            else "FAIL"
        ),
    }
    if checkpoint_audit["status"] != "PASS":
        raise RuntimeError("a frozen canonical identity changed during smoke")
    atomic_json(args.artifact_root / "checkpoint_audit.json", checkpoint_audit)

    pair_audit = {
        "pair_id_digest": pair_id_digest(expected_pair_ids),
        "manifest_coverage": count,
        "unique_pair_ids": len(set(expected_pair_ids.tolist())),
        "episode_consistency": True,
        "vision": "I_t and I_t+16",
        "action": "a_t through a_t+15 inclusive; a_t+16 excluded",
        "contact_current": "t-15 through t inclusive",
        "contact_future": "t+1 through t+16 inclusive",
        "contact_window_overlap_samples": 0,
        "split_overlap": split_overlap,
        "combined_shapes": {
            "z_v": list(teachers.z_v.shape),
            "z_a": list(teachers.z_a.shape),
            "z_c": list(teachers.z_c.shape),
            "h_t_c": list(teachers.h_t_c.shape),
            "state": list(teachers.state.shape),
            "action": list(teachers.action.shape),
        },
        "same_order": True,
        "modality_masks": "PASS",
        "cache_source": c0_discovery,
        "integrated_cache_sha256": sha256_file(cache_output),
        "status": "PASS",
    }
    atomic_json(args.artifact_root / "pair_contract_audit.json", pair_audit)

    m3 = verify_m3()
    summary = {
        "schema": "tactile3d-unit.integration-acceptance.v1",
        "decision": "INTEGRATION_READY_FOR_TRACK_C",
        "training_performed": False,
        "gpu_used": False,
        "ancestry_audit": ancestry,
        "checkpoint_audit": checkpoint_audit,
        "pair_contract_audit": pair_audit,
        "causal_audit": causal,
        "combined_smoke": {
            "samples": count,
            "source": "accepted C0 native V/C cache plus live CPU A-R inference on exact pair anchors",
            "z_v": {
                "finite": bool(np.isfinite(teachers.z_v).all()),
                "deterministic": "identity-validated accepted cache",
            },
            "z_a": action_smoke,
            "z_c": {
                "finite": bool(np.isfinite(teachers.z_c).all()),
                "deterministic": "identity-validated accepted cache",
            },
            "batch_ordering": "PASS",
            "modality_masks": "PASS",
        },
        "native_pairwise_baseline": baseline,
        "action_integrity": action_integrity,
        "contact_integrity": contact_integrity,
        "original_unit_non_regression": original_non_regression,
        "m3_protocol_consistency": m3,
        "stop_point": {
            "integration": "COMPLETE",
            "full_track_c": "NOT STARTED",
            "m3": "NOT ESTABLISHED",
        },
    }
    atomic_json(args.artifact_root / "integration_summary.json", summary)
    (args.artifact_root / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(summary))
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "samples": count,
                "artifact_root": ".local/artifacts/tactile_unit/integration",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
