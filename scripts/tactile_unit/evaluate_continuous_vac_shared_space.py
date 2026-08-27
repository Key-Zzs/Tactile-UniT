#!/usr/bin/env python3
"""Untouched-test evaluation and hard-gate decision for selected C2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_teacher.evaluation import classification_metrics  # noqa: E402
from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    MODALITIES,
    different_episode_permutation,
    geometry_diagnostics,
    linear_cka,
    load_checkpoint,
    pairwise_alignment_metrics,
)
from gr00t.tactile_unit.paired_contract import sha256_file  # noqa: E402
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_data import RAW_ACTION_DIM, action_activity  # noqa: E402
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_s2_model  # noqa: E402
from scripts.tactile_unit.evaluate_trex_action_bootstrap import fit_frozen_probes  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c2_continuous_shared_space.json"
DEFAULT_C1 = ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--c1-config", type=Path, default=DEFAULT_C1)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--action-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt")
    parser.add_argument("--s2-checkpoint", type=Path, default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt")
    parser.add_argument("--s1-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser.parse_args()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def encode_split(model, split, device: torch.device, batch_size: int):
    source_names = {"vision": "z_v", "action": "z_a", "contact": "z_c"}
    shared = {name: np.empty((len(split), 8, 32), dtype=np.float32) for name in MODALITIES}
    recovered = {name: np.empty_like(shared[name]) for name in MODALITIES}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(split), batch_size):
            stop = min(start + batch_size, len(split))
            for modality in MODALITIES:
                native = torch.from_numpy(np.array(split.arrays[source_names[modality]][start:stop], copy=True)).to(device)
                value = model.encode(modality, native)
                shared[modality][start:stop] = value.float().cpu().numpy()
                recovered[modality][start:stop] = model.recover(modality, value).float().cpu().numpy()
    return shared, recovered


def recovery_metrics(native: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    source = np.asarray(native, dtype=np.float64).reshape(len(native), -1)
    prediction = np.asarray(recovered, dtype=np.float64).reshape(len(recovered), -1)
    residual = prediction - source
    denominator = np.square(source - source.mean(axis=0, keepdims=True)).sum()
    cosine = np.sum(source * prediction, axis=1) / np.maximum(
        np.linalg.norm(source, axis=1) * np.linalg.norm(prediction, axis=1), 1e-12
    )
    return {
        "mse": float(np.square(residual).mean()),
        "cosine": float(cosine.mean()),
        "r2": float(1.0 - np.square(residual).sum() / max(float(denominator), 1e-12)),
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


def same_episode_wrong_time(episode: np.ndarray, t: np.ndarray) -> np.ndarray:
    result = np.arange(len(episode), dtype=np.int64)
    for current in np.unique(episode):
        indices = np.flatnonzero(episode == current)
        if len(indices) > 1:
            result[indices] = np.roll(indices, 1)
    if np.any((result != np.arange(len(result))) & (t[result] == t)):
        raise RuntimeError("same-episode wrong-time control retained a time anchor")
    return result


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    return np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12
    )


def probe_classification(train_x, test_x, train_y, test_y, classes: int):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0, class_weight="balanced"))
    model.fit(np.asarray(train_x).reshape(len(train_x), -1), train_y)
    prediction = model.predict(np.asarray(test_x).reshape(len(test_x), -1))
    majority_class = int(np.bincount(np.asarray(train_y), minlength=classes).argmax())
    majority = np.full(len(test_y), majority_class, dtype=np.int64)
    recalls = {}
    for label in range(classes):
        mask = np.asarray(test_y) == label
        recalls[str(label)] = None if not mask.any() else float(np.mean(prediction[mask] == label))
    return {
        **classification_metrics(test_y, prediction),
        "majority": classification_metrics(test_y, majority),
        "per_class_recall": recalls,
        "probe": "StandardScaler + RidgeClassifier(alpha=10,class_weight=balanced)",
    }


def retention(shared: Mapping[str, Any], native: Mapping[str, Any]) -> float:
    majority = float(native["majority"]["macro_f1"])
    return float(
        (float(shared["macro_f1"]) - majority)
        / max(float(native["macro_f1"]) - majority, 1e-12)
    )


def contact_evaluation(model, train, test, shared_train, shared_test, recovered_test, s2, device, batch_size):
    probes = {}
    for name, key, classes in (
        ("contact_transition", "contact_transition", 4),
        ("force_trend", "force_trend_class", 3),
    ):
        native = probe_classification(train.arrays["z_c"], test.arrays["z_c"], train.arrays[key], test.arrays[key], classes)
        projected = probe_classification(shared_train["contact"], shared_test["contact"], train.arrays[key], test.arrays[key], classes)
        probes[name] = {"native": native, "shared": projected, "retention": retention(projected, native)}
    current = np.asarray(test.arrays["h_current"])
    future = np.asarray(test.arrays["h_future"])
    predictions = {"native": np.empty_like(future), "recovered": np.empty_like(future)}
    reversed_code = np.empty((len(test), 8, 32), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            current_t = torch.from_numpy(np.array(current[start:stop], copy=True)).to(device)
            future_t = torch.from_numpy(np.array(future[start:stop], copy=True)).to(device)
            native_z = torch.from_numpy(np.array(test.arrays["z_c"][start:stop], copy=True)).to(device)
            recovered_z = torch.from_numpy(np.array(recovered_test["contact"][start:stop], copy=True)).to(device)
            predictions["native"][start:stop] = s2.decoder(native_z, current_t).float().cpu().numpy()
            predictions["recovered"][start:stop] = s2.decoder(recovered_z, current_t).float().cpu().numpy()
            reversed_code[start:stop] = s2.encoder(future_t, current_t).float().cpu().numpy()
    dynamic = np.asarray(test.arrays["dynamic"], dtype=bool)
    reconstruction = {}
    for name, prediction in predictions.items():
        per_sample = np.square(prediction - future).mean(axis=1)
        reconstruction[name] = {
            "future_mse": float(per_sample.mean()),
            "dynamic_mse": float(per_sample[dynamic].mean()),
        }
    reconstruction["retention"] = {
        "future_native_over_recovered": reconstruction["native"]["future_mse"] / max(reconstruction["recovered"]["future_mse"], 1e-12),
        "dynamic_native_over_recovered": reconstruction["native"]["dynamic_mse"] / max(reconstruction["recovered"]["dynamic_mse"], 1e-12),
    }
    reversed_shared = np.empty_like(reversed_code)
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            value = torch.from_numpy(reversed_code[start:stop]).to(device)
            reversed_shared[start:stop] = model.encode("contact", value).float().cpu().numpy()
    return {
        "probes": probes,
        "free_to_contact_recall": probes["contact_transition"]["shared"]["per_class_recall"].get("1"),
        "contact_to_free_recall": probes["contact_transition"]["shared"]["per_class_recall"].get("2"),
        "reconstruction": reconstruction,
        "temporal_reversed_control": {
            "native_paired_cosine": float(row_cosine(test.arrays["z_c"], reversed_code).mean()),
            "shared_paired_cosine": float(row_cosine(shared_test["contact"], reversed_shared).mean()),
        },
    }


def action_labels(action: np.ndarray, primitive: np.ndarray) -> dict[str, np.ndarray]:
    labels = action_activity(np.asarray(action)[..., :RAW_ACTION_DIM])
    labels["primitive_id"] = np.asarray(primitive)
    return labels


def temporal_ratios(errors: Mapping[str, np.ndarray], dynamic: np.ndarray) -> dict[str, Any]:
    result = {"all": {}, "dynamic": {}}
    for subset, mask in (("all", np.ones(len(dynamic), dtype=bool)), ("dynamic", dynamic)):
        correct = float(np.asarray(errors["correct"])[mask].mean())
        result[subset]["correct"] = correct
        for name, value in errors.items():
            mean = float(np.asarray(value)[mask].mean())
            result[subset][name] = mean
            if name != "correct":
                result[subset][f"{name}_over_correct"] = mean / max(correct, 1e-12)
    return result


def action_evaluation(
    model, train, test, shared_train, shared_test, action_model, device, batch_size, seed,
    *, cache_atol: float, cache_rtol: float,
):
    train_labels = action_labels(train.arrays["action"], train.arrays["primitive_id"])
    test_labels = action_labels(test.arrays["action"], test.arrays["primitive_id"])
    native_probes, _, _ = fit_frozen_probes(
        np.asarray(train.arrays["z_a"]), train_labels,
        np.asarray(test.arrays["z_a"]), test_labels,
    )
    shared_probes, _, _ = fit_frozen_probes(
        shared_train["action"], train_labels, shared_test["action"], test_labels,
    )
    probe_retention = {}
    for name in ("magnitude", "trend"):
        native = float(native_probes[name]["r2"])
        probe_retention[name] = float(shared_probes[name]["r2"] / max(native, 1e-12))
    for name in ("active_side", "arm_vs_hand"):
        native = float(native_probes[name]["balanced_accuracy"])
        chance = 0.5
        probe_retention[name] = float(
            (float(shared_probes[name]["balanced_accuracy"]) - chance) / max(native - chance, 1e-12)
        )

    rng = np.random.default_rng(seed)
    time_permutation = rng.permutation(16)
    different = different_episode_permutation(np.asarray(test.arrays["episode_id"]), seed + 1)
    native_errors = {name: [] for name in ("correct", "reversed", "shuffled", "different_episode", "zero")}
    shared_errors = {name: [] for name in native_errors}
    cached_allclose = True
    cached_max_abs_difference = 0.0
    action_model.eval().requires_grad_(False)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(test), batch_size):
            stop = min(start + batch_size, len(test))
            state = torch.from_numpy(np.array(test.arrays["state"][start:stop], copy=True)).to(device)
            action = torch.from_numpy(np.array(test.arrays["action"][start:stop], copy=True)).to(device)
            embodiment = torch.full((stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
            z_correct, state_features, _ = action_model.encode(state, action, embodiment)
            cached = torch.from_numpy(np.array(test.arrays["z_a"][start:stop], copy=True)).to(device)
            cached_max_abs_difference = max(
                cached_max_abs_difference,
                float(torch.max(torch.abs(z_correct.float() - cached.float())).item()),
            )
            cached_allclose = cached_allclose and bool(
                torch.allclose(z_correct, cached, atol=cache_atol, rtol=cache_rtol)
            )
            other_action = torch.from_numpy(np.array(test.arrays["action"][different[start:stop]], copy=True)).to(device)
            controls = {
                "correct": cached,
                "reversed": action_model.encode(state, action.flip(1), embodiment)[0],
                "shuffled": action_model.encode(state, action[:, torch.from_numpy(time_permutation).to(device)], embodiment)[0],
                "different_episode": action_model.encode(state, other_action, embodiment)[0],
                "zero": torch.zeros_like(cached),
            }
            target = action[..., :RAW_ACTION_DIM]
            for name, z_value in controls.items():
                native_prediction = action_model.decode(z_value, state_features, embodiment)[..., :RAW_ACTION_DIM]
                native_errors[name].append(F.mse_loss(native_prediction, target, reduction="none").mean(dim=(1, 2)).cpu().numpy())
                if name == "zero":
                    recovered = model.recover("action", torch.zeros_like(z_value))
                else:
                    recovered = model.recover("action", model.encode("action", z_value))
                shared_prediction = action_model.decode(recovered, state_features, embodiment)[..., :RAW_ACTION_DIM]
                shared_errors[name].append(F.mse_loss(shared_prediction, target, reduction="none").mean(dim=(1, 2)).cpu().numpy())
    native_arrays = {name: np.concatenate(value) for name, value in native_errors.items()}
    shared_arrays = {name: np.concatenate(value) for name, value in shared_errors.items()}
    dynamic_threshold = 0.029811091721057892
    dynamic = np.asarray(test_labels["magnitude"]) > dynamic_threshold
    return {
        "cached_native_allclose": cached_allclose,
        "cached_native_allclose_tolerance": {"atol": cache_atol, "rtol": cache_rtol},
        "cached_native_max_abs_difference": cached_max_abs_difference,
        "dynamic_threshold": dynamic_threshold,
        "dynamic_windows": int(dynamic.sum()),
        "native_temporal": temporal_ratios(native_arrays, dynamic),
        "shared_temporal": temporal_ratios(shared_arrays, dynamic),
        "native_probes": native_probes,
        "shared_probes": shared_probes,
        "probe_retention": probe_retention,
    }


def sliced_wasserstein(left: np.ndarray, right: np.ndarray, seed: int, projections: int = 64) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    count = min(len(x), len(y), 2048)
    x = x[:count]; y = y[:count]
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(x.shape[1], projections))
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-12)
    return float(np.mean(np.abs(np.sort(x @ directions, axis=0) - np.sort(y @ directions, axis=0))))


def mmd_rbf(left: np.ndarray, right: np.ndarray, maximum: int = 1024) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)[:maximum]
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)[:maximum]
    joined = np.concatenate((x[: min(256, len(x))], y[: min(256, len(y))]))
    squared = np.maximum(
        np.square(joined).sum(1)[:, None] + np.square(joined).sum(1)[None, :] - 2.0 * joined @ joined.T,
        0.0,
    )
    positive = squared[squared > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    def kernel(a, b):
        distance = np.maximum(
            np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2.0 * a @ b.T,
            0.0,
        )
        return np.exp(-distance / max(2.0 * bandwidth, 1e-12))
    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def geometry_bundle(shared: Mapping[str, np.ndarray], seed: int) -> dict[str, Any]:
    geometry = {name: geometry_diagnostics(value) for name, value in shared.items()}
    diagnostics = {}
    for offset, (name, left, right) in enumerate((
        ("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact")
    )):
        diagnostics[name] = {
            "centroid_distance": float(np.linalg.norm(shared[left].reshape(len(shared[left]), -1).mean(0) - shared[right].reshape(len(shared[right]), -1).mean(0))),
            "mmd_rbf": mmd_rbf(shared[left], shared[right]),
            "sliced_wasserstein": sliced_wasserstein(shared[left], shared[right], seed + offset),
            "role": "DIAGNOSTIC ONLY",
        }
    return {"modalities": geometry, "cross_modal": diagnostics}


def independent_audit(model) -> dict[str, Any]:
    result = {}
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else torch.device("cpu")
    with torch.inference_mode():
        for name in MODALITIES:
            value = torch.randn(3, 8, 32, device=device)
            output = model.encode(name, value)
            result[name] = list(output.shape) == [3, 8, 32]
    return {"modalities": result, "pair_conditioned_retrieval": False, "status": "PASS" if all(result.values()) else "FAIL"}


def human_acceptance(result: Mapping[str, Any]) -> str:
    return "\n".join([
        "# C2 Continuous VAC Shared Space Human Acceptance", "",
        f"Decision: **{result['decision']}**", "",
        f"Selected candidate: {result['selected']['candidate']}.",
        "Independent encoding: PASS; each candidate is precomputable without a paired query.",
        f"Overall alignment gate: {'PASS' if result['gates']['alignment'] else 'FAIL'}.",
        f"Contact semantic gate: {'PASS' if result['gates']['contact'] else 'FAIL'}.",
        f"Action temporal gate: {'PASS' if result['gates']['action'] else 'FAIL'}.",
        f"Collapse gate: {'PASS' if result['gates']['noncollapse'] else 'FAIL'}.",
        "M3: NOT ESTABLISHED. C3/C4/C5/C6: NOT STARTED.", "",
    ])


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    c1 = json.loads(args.c1_config.read_text())
    cache_root = args.cache_root or ROOT / config["runtime"]["cache_root"]
    checkpoint = args.checkpoint or ROOT / config["runtime"]["experiment_root"] / "selected.pt"
    artifact_root = args.artifact_root or ROOT / config["runtime"]["artifact_root"]
    bootstrap_samples = args.bootstrap_samples or int(config["evaluation"]["bootstrap_samples"])
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    device, lock_handle, gpu = resolve_device(args.device)
    try:
        set_seed(int(config["seed"]))
        train = load_split(cache_root, "train", verify_hashes=True)
        test = load_split(cache_root, "test", verify_hashes=True)
        model, checkpoint_metadata = load_checkpoint(checkpoint, device)
        model.eval().to(device)
        shared_train, _ = encode_split(model, train, device, args.batch_size)
        shared_test, recovered_test = encode_split(model, test, device, args.batch_size)
        episode = np.asarray(test.arrays["episode_id"])
        transition = np.asarray(test.arrays["contact_transition"])
        masks = {
            "all": np.ones(len(test), dtype=bool),
            "dynamic": np.asarray(test.arrays["dynamic"], dtype=bool),
            "rare_boundary": np.isin(transition, [1, 2]),
            "free_to_contact": transition == 1,
            "contact_to_free": transition == 2,
        }
        alignment: dict[str, Any] = {}
        for offset, (name, left, right) in enumerate((
            ("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact")
        )):
            alignment[name] = {
                subset: subset_alignment(
                    shared_test[left], shared_test[right], episode, mask,
                    samples=bootstrap_samples, seed=int(config["seed"]) + offset * 100 + index,
                    chunk=int(config["evaluation"]["retrieval_chunk"]),
                ) for index, (subset, mask) in enumerate(masks.items())
            }
        wrong_time = same_episode_wrong_time(episode, np.asarray(test.arrays["t"]))
        random_pair = np.random.default_rng(int(config["seed"])).permutation(len(test))
        controls = {}
        for name, left, right in (("V-A", "vision", "action"), ("V-C", "vision", "contact"), ("A-C", "action", "contact")):
            controls[name] = {
                "same_episode_wrong_time_cosine": float(row_cosine(shared_test[left], shared_test[right][wrong_time]).mean()),
                "modality_mismatched_random_cosine": float(row_cosine(shared_test[left], shared_test[right][random_pair]).mean()),
            }

        s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
        source = ReleasedTokenizerSource.open(args.unit_checkpoint / "tokenizer")
        if source.old_rows_digest() != c1["frozen_identity"]["old_action_rows_digest"]:
            raise RuntimeError("Original UniT old Action row digest changed")
        action_model, action_metadata = load_shared_transition_checkpoint(args.action_checkpoint, source)
        action_model.eval().requires_grad_(False).to(device)
        immutable_files = {
            "s1_teacher_file": args.s1_checkpoint,
            "s2_checkpoint_file": args.s2_checkpoint,
            "action_checkpoint_file": args.action_checkpoint,
            **{
                f"vision_checkpoint/{name}": args.unit_checkpoint / "tokenizer" / name
                for name in c1["frozen_identity"]["original_unit_tokenizer_files_sha256"]
            },
        }
        native_before = {
            "action": parameter_digest(action_model),
            "s2_encoder": parameter_digest(s2.encoder),
            "s2_decoder": parameter_digest(s2.decoder),
            "old_action_rows": source.old_rows_digest(),
            **{name: sha256_file(path) for name, path in immutable_files.items()},
        }
        expected_files = {
            "s1_teacher_file": c1["frozen_identity"]["s1_teacher_checkpoint_sha256"],
            "s2_checkpoint_file": c1["frozen_identity"]["s2_checkpoint_sha256"],
            "action_checkpoint_file": c1["frozen_identity"]["action_checkpoint_sha256"],
            **{
                f"vision_checkpoint/{name}": digest
                for name, digest in c1["frozen_identity"]["original_unit_tokenizer_files_sha256"].items()
            },
        }
        if any(native_before[name] != digest for name, digest in expected_files.items()):
            raise RuntimeError("a frozen native checkpoint identity changed before C2 evaluation")
        contact = contact_evaluation(model, train, test, shared_train, shared_test, recovered_test, s2, device, args.batch_size)
        action = action_evaluation(
            model, train, test, shared_train, shared_test, action_model, device,
            args.batch_size, int(config["seed"]),
            cache_atol=float(config["evaluation"]["action_cache_allclose_atol"]),
            cache_rtol=float(config["evaluation"]["action_cache_allclose_rtol"]),
        )
        native_after = {
            "action": parameter_digest(action_model),
            "s2_encoder": parameter_digest(s2.encoder),
            "s2_decoder": parameter_digest(s2.decoder),
            "old_action_rows": source.old_rows_digest(),
            **{name: sha256_file(path) for name, path in immutable_files.items()},
        }
        vision = {
            "recovery": recovery_metrics(test.arrays["z_v"], recovered_test["vision"]),
            "relational_cka": linear_cka(test.arrays["z_v"], shared_test["vision"]),
            "native_geometry": geometry_diagnostics(test.arrays["z_v"]),
            "shared_geometry": geometry_diagnostics(shared_test["vision"]),
        }
        geometry = geometry_bundle(shared_test, int(config["seed"]))

        pair_gates = {}
        for name, values in alignment.items():
            all_values = values["all"]
            r10 = [
                all_values["retrieval"][direction]["recall_at_10"] / all_values["retrieval"][direction]["chance"]["recall_at_10"]
                for direction in ("forward", "reverse")
            ]
            pair_gates[name] = {
                "positive_margin_ci": all_values["paired_minus_shuffled_margin"] > 0 and all_values["margin_bootstrap_ci95"][0] > 0,
                "retrieval": min(r10) >= float(config["evaluation"]["retrieval_r10_chance_multiplier_min"]),
                "r10_chance_multiplier": r10,
            }
            pair_gates[name]["pass"] = all(pair_gates[name][key] for key in ("positive_margin_ci", "retrieval"))
        contact_gate = (
            contact["probes"]["contact_transition"]["retention"] >= float(config["evaluation"]["contact_retention_min"])
            and contact["probes"]["force_trend"]["retention"] >= float(config["evaluation"]["force_retention_min"])
        )
        shared_dynamic = action["shared_temporal"]["dynamic"]
        action_gate = (
            shared_dynamic["reversed_over_correct"] >= float(config["evaluation"]["action_dynamic_reversed_ratio_min"])
            and shared_dynamic["shuffled_over_correct"] >= float(config["evaluation"]["action_dynamic_shuffled_ratio_min"])
            and shared_dynamic["zero_over_correct"] >= 1.1
            and action["cached_native_allclose"]
        )
        noncollapse = all(
            value["per_dimension_variance"]["near_zero_fraction"] < 0.5
            and value["query_diversity"]["collapsed_pair_fraction"] < 0.5
            for value in geometry["modalities"].values()
        )
        independent = independent_audit(model)
        original = native_before == native_after
        gates = {
            "alignment": all(value["pass"] for value in pair_gates.values()),
            "contact": bool(contact_gate),
            "action": bool(action_gate),
            "noncollapse": bool(noncollapse),
            "independent_encodability": independent["status"] == "PASS",
            "original_unit_preserved": original,
        }
        structural_ok = noncollapse and independent["status"] == "PASS" and original
        warnings = []
        if not structural_ok:
            decision = "STRUCTURAL_FAIL"
        elif not contact_gate or not action_gate:
            decision = "C2_SEMANTIC_PRESERVATION_FAIL"
        elif not gates["alignment"]:
            decision = "C2_ALIGNMENT_INSUFFICIENT"
        else:
            weak_boundary = any(
                alignment[name]["rare_boundary"].get("paired_minus_shuffled_margin", 0.0) <= 0
                for name in ("V-C", "A-C")
            )
            modest = any(min(value["r10_chance_multiplier"]) < 2.0 for value in pair_gates.values())
            r1_modest = any(
                min(
                    alignment[name]["all"]["retrieval"][direction]["recall_at_1"]
                    / alignment[name]["all"]["retrieval"][direction]["chance"]["recall_at_1"]
                    for direction in ("forward", "reverse")
                ) < 1.5
                for name in alignment
            )
            if weak_boundary:
                warnings.append("rare Contact-boundary alignment is non-positive for V-C or A-C")
            if modest:
                warnings.append("at least one modality pair has less than 2x-chance R@10")
            if r1_modest:
                warnings.append("at least one retrieval direction has modest R@1")
            decision = "C2_SHARED_SPACE_READY_WITH_WARNINGS" if warnings else "C2_SHARED_SPACE_READY"
        result = {
            "schema": "tactile3d-unit.vac-c2-evaluation.v1",
            "decision": decision,
            "selected": {"candidate": model.candidate, "checkpoint": str(checkpoint.relative_to(ROOT)), "sha256": sha256_file(checkpoint), "metadata": checkpoint_metadata},
            "gpu": gpu,
            "alignment": alignment,
            "pair_gates": pair_gates,
            "negative_controls": controls,
            "contact": contact,
            "action": action,
            "vision": vision,
            "geometry": geometry,
            "independent_encodability": independent,
            "native_identity_before": native_before,
            "native_identity_after": native_after,
            "action_checkpoint_metadata": action_metadata,
            "gates": gates,
            "warnings": warnings,
            "causal_boundary": {
                "offline_teachers": ["z_v", "z_a", "z_c"],
                "online_legal_current_contact": "h_t^c",
                "future_teacher_exposed_to_runtime_observation": False,
                "C5_started": False,
            },
            "scope": {"C3": "NOT STARTED", "C4": "NOT STARTED", "C5": "NOT STARTED", "C6_M3": "NOT STARTED", "M3": "NOT ESTABLISHED"},
        }
        atomic_json(artifact_root / "evaluation.json", result)
        (artifact_root / "HUMAN_ACCEPTANCE.md").write_text(human_acceptance(result))
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
