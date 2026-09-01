#!/usr/bin/env python3
"""Audit frozen C2/C3 Contact representations on a train-internal DS split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.contact_dynamics.evaluation import (  # noqa: E402
    different_episode_permutation,
    query_diversity,
)
from gr00t.contact_dynamics.models import (  # noqa: E402
    ContactDynamicsEncoder,
    ContactDynamicsModel,
    DeltaMLPEncoder,
    LatentTransitionDecoder,
)
from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402


PAIR_PATH = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics/train.npz"
PAIR_SOURCE_PATH = ROOT / ".local/cache/simulation/s4_2/pairs/train.npz"
C2_PATH = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/C2/best.pt"
C3_PATH = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2ds"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str) -> ContactDynamicsModel:
    encoder = DeltaMLPEncoder() if name == "C2" else ContactDynamicsEncoder()
    return ContactDynamicsModel(encoder, LatentTransitionDecoder())


def load_model(path: Path, device: torch.device) -> tuple[ContactDynamicsModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, checkpoint


@torch.inference_mode()
def encode(
    model: ContactDynamicsModel,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(current), 8, 32), dtype=np.float32)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        result[start:stop] = (
            model.encoder(
                torch.from_numpy(np.array(current[start:stop], copy=True)).to(device),
                torch.from_numpy(np.array(future[start:stop], copy=True)).to(device),
            )
            .float()
            .cpu()
            .numpy()
        )
    return result


@torch.inference_mode()
def decode(
    model: ContactDynamicsModel,
    code: np.ndarray,
    current: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty_like(current)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        result[start:stop] = (
            model.decoder(
                torch.from_numpy(np.array(code[start:stop], copy=True)).to(device),
                torch.from_numpy(np.array(current[start:stop], copy=True)).to(device),
            )
            .float()
            .cpu()
            .numpy()
        )
    return result


def group_split(task: np.ndarray, source: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    groups = sorted({(str(t), str(s)) for t, s in zip(task, source)})
    dev_groups: set[tuple[str, str]] = set()
    by_task: dict[str, list[tuple[str, str]]] = {}
    for group in groups:
        by_task.setdefault(group[0], []).append(group)
    for task_name, task_groups in sorted(by_task.items()):
        ranked = sorted(
            task_groups,
            key=lambda value: hashlib.sha256(
                f"{seed}:{value[0]}:{value[1]}".encode("utf-8")
            ).hexdigest(),
        )
        dev_count = max(1, int(round(0.20 * len(ranked))))
        dev_groups.update(ranked[:dev_count])
    row_groups = np.asarray([(str(t), str(s)) in dev_groups for t, s in zip(task, source)])
    train_indices = np.flatnonzero(~row_groups)
    dev_indices = np.flatnonzero(row_groups)
    manifest = {
        "schema": "tactile3d-unit.s4-2ds-split-manifest.v1",
        "seed": seed,
        "source": ".local/cache/simulation/s4_2r/contact_dynamics/train.npz",
        "group_key": ["task", "source_trajectory_id"],
        "policy": "per-task deterministic SHA256 ordering; nearest whole-group 80/20 split",
        "original_train_groups": len(groups),
        "ds_train_groups": len(groups) - len(dev_groups),
        "ds_dev_groups": len(dev_groups),
        "ds_train_pairs": len(train_indices),
        "ds_dev_pairs": len(dev_indices),
        "ds_train_group_fraction": (len(groups) - len(dev_groups)) / len(groups),
        "ds_dev_group_fraction": len(dev_groups) / len(groups),
        "ds_train_groups_by_task": {
            name: sum(group not in dev_groups for group in values)
            for name, values in sorted(by_task.items())
        },
        "ds_dev_groups_by_task": {
            name: sum(group in dev_groups for group in values)
            for name, values in sorted(by_task.items())
        },
        "ds_dev_group_ids": [list(value) for value in sorted(dev_groups)],
        "group_overlap": 0,
        "formal_validation_excluded": True,
        "formal_test_excluded": True,
        "formal_test_model_metrics_loaded": False,
    }
    return train_indices, dev_indices, manifest


def classification_probe(
    train_x: np.ndarray,
    dev_x: np.ndarray,
    train_y: np.ndarray,
    dev_y: np.ndarray,
    classes: list[int],
) -> tuple[dict[str, Any], np.ndarray]:
    model = RidgeClassifier(alpha=10.0).fit(train_x, train_y)
    prediction = model.predict(dev_x)
    recalls = recall_score(dev_y, prediction, labels=classes, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(dev_y, prediction, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(dev_y, prediction)),
        "per_class_recall": {str(label): float(value) for label, value in zip(classes, recalls)},
        "train_classes": np.unique(train_y).astype(int).tolist(),
        "dev_classes": np.unique(dev_y).astype(int).tolist(),
        "probe": "RidgeClassifier(alpha=10.0)",
    }, prediction


def regression_probe(
    train_x: np.ndarray,
    dev_x: np.ndarray,
    train_y: np.ndarray,
    dev_y: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    model = Ridge(alpha=10.0).fit(train_x, train_y)
    prediction = model.predict(dev_x)
    return {
        "r2": float(r2_score(dev_y, prediction)),
        "mae": float(mean_absolute_error(dev_y, prediction)),
        "probe": "Ridge(alpha=10.0)",
    }, prediction


def probe_suite(
    train_code: np.ndarray,
    dev_code: np.ndarray,
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    dev_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_x = train_code[train_indices].reshape(len(train_indices), -1)
    dev_x = dev_code.reshape(len(dev_indices), -1)
    result: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, key, classes in (
        ("contact_transition", "contact_transition", [0, 1, 2, 3]),
        ("force_trend", "force_trend", [0, 1, 2]),
    ):
        result[name], predictions[name] = classification_probe(
            train_x,
            dev_x,
            arrays[key][train_indices],
            arrays[key][dev_indices],
            classes,
        )
    regions = []
    for region in range(5):
        metrics, prediction = classification_probe(
            train_x,
            dev_x,
            arrays["region_contact_change"][train_indices, region],
            arrays["region_contact_change"][dev_indices, region],
            [0, 1, 2],
        )
        regions.append({"region": region, **metrics})
        predictions[f"region_{region}"] = prediction
    result["per_region_contact_change"] = regions
    result["future_force_magnitude"], predictions["future_force_magnitude"] = regression_probe(
        train_x,
        dev_x,
        arrays["future_total_force"][train_indices],
        arrays["future_total_force"][dev_indices],
    )
    return result, predictions


def spectrum_metrics(code: np.ndarray, seed: int) -> dict[str, Any]:
    flat = np.asarray(code, dtype=np.float64).reshape(len(code), -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = np.var(flat, axis=0)
    eigen = np.square(singular) / max(len(flat) - 1, 1)
    probability = eigen / max(float(eigen.sum()), 1e-12)
    cumulative = np.cumsum(probability)
    effective_rank = float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-15)))))
    participation = float(np.square(eigen.sum()) / max(float(np.square(eigen).sum()), 1e-12))
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(flat), size=min(512, len(flat)), replace=False)
    sampled = flat[subset]
    distance = np.linalg.norm(sampled[:, None] - sampled[None, :], axis=-1)
    qd = query_diversity(code)
    return {
        "native_latent_dim": 256,
        "common_shape": [8, 32],
        "compression_ratio_pair_input_to_latent": 2.0,
        "effective_rank": effective_rank,
        "stable_rank": float(np.square(singular).sum() / max(float(np.square(singular[0])), 1e-12)),
        "participation_ratio": participation,
        "top_pc_fraction": float(probability[0]),
        "d90": int(np.searchsorted(cumulative, 0.90) + 1),
        "d95": int(np.searchsorted(cumulative, 0.95) + 1),
        "d99": int(np.searchsorted(cumulative, 0.99) + 1),
        "mean_variance": float(variance.mean()),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
        "sampled_pairwise_distance_mean": float(distance[np.triu_indices(len(sampled), 1)].mean()),
        "token_norm_mean": float(np.linalg.norm(code, axis=2).mean()),
        "query_diversity": qd,
        "spectrum_fraction": probability.astype(float).tolist(),
    }


def per_sample_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(prediction.astype(np.float64) - target.astype(np.float64)).mean(axis=1)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        stop = min(samples, start + 100)
        index = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[index].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).astype(float).tolist()


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = left.reshape(len(left), -1).astype(np.float64)
    y = right.reshape(len(right), -1).astype(np.float64)
    return np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12
    )


def control_suite(
    model: ContactDynamicsModel,
    code: np.ndarray,
    arrays: dict[str, np.ndarray],
    dev_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    current = arrays["h_current"][dev_indices]
    future = arrays["h_future"][dev_indices]
    episode = arrays["episode_id"][dev_indices]
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(len(dev_indices))
    different_index = different_episode_permutation(episode, seed=seed)
    reversed_code = encode(model, future, current, device, batch_size)
    mismatched_code = encode(model, current, future[different_index], device, batch_size)
    mean = code.mean(axis=0, keepdims=True)
    std = code.std(axis=0, keepdims=True)
    gaussian = rng.normal(size=code.shape).astype(np.float32) * std + mean
    codes = {
        "full": code,
        "zero": np.zeros_like(code),
        "shuffled": code[shuffled_index],
        "different_episode": code[different_index],
        "gaussian_matched_variance": gaussian,
        "reversed": reversed_code,
        "mismatched_future": mismatched_code,
    }
    errors = {name: per_sample_mse(decode(model, value, current, device, batch_size), future) for name, value in codes.items()}
    full = errors["full"]
    controls = {
        name: {
            "future_mse": float(error.mean()),
            "over_full_ratio": float(error.mean() / max(float(full.mean()), 1e-12)),
            "paired_error_increase_ci95": bootstrap_ci(error - full, bootstrap_samples, seed + offset),
        }
        for offset, (name, error) in enumerate(errors.items())
    }
    invalid = ("zero", "shuffled", "different_episode", "gaussian_matched_variance")
    necessity_pass = all(
        controls[name]["future_mse"] > controls["full"]["future_mse"]
        and controls[name]["paired_error_increase_ci95"][0] > 0.0
        for name in invalid
    )
    temporal_pass = all(
        controls[name]["future_mse"] > controls["full"]["future_mse"]
        and controls[name]["paired_error_increase_ci95"][0] > 0.0
        for name in ("reversed", "different_episode", "mismatched_future")
    )
    result = {
        "decoder_recovery": controls,
        "latent_cosine_to_correct": {
            name: float(row_cosine(code, value).mean()) for name, value in codes.items()
        },
        "latent_l2_to_correct": {
            name: float(np.linalg.norm(code.reshape(len(code), -1) - value.reshape(len(value), -1), axis=1).mean())
            for name, value in codes.items()
        },
        "information_necessity": "PASS" if necessity_pass else "FAIL",
        "transition_specificity": "PASS" if temporal_pass else "FAIL",
        "raw_pair_recomputed": ["reversed", "mismatched_future"],
    }
    return result, errors


def f1_difference_ci(
    target: np.ndarray,
    c2_prediction: np.ndarray,
    c3_prediction: np.ndarray,
    classes: list[int],
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        rows = rng.integers(0, len(target), size=len(target))
        values[index] = f1_score(
            target[rows], c3_prediction[rows], labels=classes, average="macro", zero_division=0
        ) - f1_score(
            target[rows], c2_prediction[rows], labels=classes, average="macro", zero_division=0
        )
    return np.quantile(values, [0.025, 0.975]).astype(float).tolist()


def stability_audit(
    path: Path,
    expected: np.ndarray,
    current: np.ndarray,
    future: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    cold, _ = load_model(path, device)
    reference = expected[:64]
    repeat = encode(cold, current[:64], future[:64], device, 64)
    batch_one = encode(cold, current[:64], future[:64], device, 1)
    order = np.arange(64)[::-1]
    reordered = encode(cold, current[:64][order], future[:64][order], device, 17)[::-1]
    differences = {
        "cold_reload": float(np.max(np.abs(reference - repeat))),
        "batch_size_change": float(np.max(np.abs(reference - batch_one))),
        "sample_order_change": float(np.max(np.abs(reference - reordered))),
    }
    tolerance = 1e-5
    return {
        "tolerance": tolerance,
        "max_absolute_differences": differences,
        "deterministic": all(value <= tolerance for value in differences.values()),
    }


def contract_audit(name: str, path: Path, checkpoint: dict[str, Any], model: ContactDynamicsModel) -> dict[str, Any]:
    encoder_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in model.decoder.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema": "tactile3d-unit.s4-2ds-latent-contract.v1",
        "candidate": name,
        "architecture": type(model.encoder).__name__ + " + LatentTransitionDecoder",
        "inputs": ["h_current [B,256]", "h_future [B,256]"],
        "encoder_input": (
            "concat(h_current, h_future-h_current)"
            if name == "C2"
            else "independent current/future/delta projections then concatenation"
        ),
        "explicit_bottleneck": True,
        "bottleneck_before_decoder": True,
        "native_latent_shape": [8, 32],
        "common_latent_shape": [8, 32],
        "adapter": "identity",
        "deterministic_by_architecture": True,
        "independently_extractable": True,
        "decoder_consumes_bottleneck": True,
        "depends_on_batch_peers": False,
        "depends_on_target_labels": False,
        "depends_on_future_metadata": False,
        "depends_on_decoder_hidden_state": False,
        "task_reward_success_leakage": False,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_schema": checkpoint["schema"],
        "checkpoint_model": checkpoint["model"],
        "encoder_parameters": encoder_parameters,
        "decoder_parameters": decoder_parameters,
        "total_parameters": total_parameters,
        "finite_checkpoint": all(torch.isfinite(value).all().item() for value in checkpoint["state_dict"].values()),
        "eligibility": "REPRESENTATION_CANDIDATE",
        "formal_test_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    arrays = load_npz(PAIR_PATH)
    source_arrays = load_npz(PAIR_SOURCE_PATH)
    for name in ("pair_id", "task", "source_trajectory_id"):
        if not np.array_equal(arrays[name], source_arrays[name]):
            raise RuntimeError(f"Contact cache identity mismatch for {name}")
    train_indices, dev_indices, split_manifest = group_split(
        arrays["task"], arrays["source_trajectory_id"], args.seed
    )
    split_manifest["source_sha256"] = sha256_file(PAIR_PATH)
    split_manifest["pair_source_sha256"] = sha256_file(PAIR_SOURCE_PATH)
    split_manifest["pair_identity_exact"] = True
    atomic_json(args.artifacts / "ds_split_manifest.json", split_manifest)

    candidates: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    errors: dict[str, dict[str, np.ndarray]] = {}
    codes: dict[str, np.ndarray] = {}
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for name, path in (("C2", C2_PATH), ("C3", C3_PATH)):
        model, checkpoint = load_model(path, device)
        code = encode(model, arrays["h_current"], arrays["h_future"], device, args.batch_size)
        codes[name] = code
        contract = contract_audit(name, path, checkpoint, model)
        stability = stability_audit(
            path,
            code[dev_indices],
            arrays["h_current"][dev_indices],
            arrays["h_future"][dev_indices],
            device,
        )
        contract["stability"] = stability
        contract["deterministic"] = stability["deterministic"]
        atomic_json(args.artifacts / f"{name.lower()}_latent_contract.json", contract)
        probes, candidate_predictions = probe_suite(
            code,
            code[dev_indices],
            arrays,
            train_indices,
            dev_indices,
        )
        geometry = spectrum_metrics(code[dev_indices], args.seed)
        controls, candidate_errors = control_suite(
            model,
            code[dev_indices],
            arrays,
            dev_indices,
            device,
            args.batch_size,
            args.seed,
            args.bootstrap_samples,
        )
        finite = bool(np.isfinite(code).all())
        no_collapse = bool(
            geometry["near_zero_variance_fraction"] <= 0.5
            and geometry["query_diversity"]["collapsed_sample_fraction"] == 0.0
        )
        hard_gates = {
            "finite": finite,
            "no_structural_collapse": no_collapse,
            "contact_transition_macro_f1_at_least_0_90": probes["contact_transition"]["macro_f1"] >= 0.90,
            "force_trend_macro_f1_at_least_0_90": probes["force_trend"]["macro_f1"] >= 0.90,
            "information_necessity_controls": controls["information_necessity"] == "PASS",
            "transition_specificity": controls["transition_specificity"] == "PASS",
            "deterministic": stability["deterministic"],
        }
        candidates[name] = {
            "contract": contract,
            "probes": probes,
            "geometry": geometry,
            "temporal_and_controls": controls,
            "hard_gates": {key: "PASS" if value else "FAIL" for key, value in hard_gates.items()},
            "contact_only_eligible": all(hard_gates.values()),
        }
        predictions[name] = candidate_predictions
        errors[name] = candidate_errors

    np.savez_compressed(
        args.cache_root / "contact_candidates.npz",
        pair_id=arrays["pair_id"],
        ds_train_indices=train_indices,
        ds_dev_indices=dev_indices,
        c2=codes["C2"],
        c3=codes["C3"],
    )
    comparison = {
        "contact_macro_f1_c3_minus_c2": candidates["C3"]["probes"]["contact_transition"]["macro_f1"]
        - candidates["C2"]["probes"]["contact_transition"]["macro_f1"],
        "contact_macro_f1_difference_ci95": f1_difference_ci(
            arrays["contact_transition"][dev_indices],
            predictions["C2"]["contact_transition"],
            predictions["C3"]["contact_transition"],
            [0, 1, 2, 3],
            args.bootstrap_samples,
            args.seed,
        ),
        "force_macro_f1_c3_minus_c2": candidates["C3"]["probes"]["force_trend"]["macro_f1"]
        - candidates["C2"]["probes"]["force_trend"]["macro_f1"],
        "force_macro_f1_difference_ci95": f1_difference_ci(
            arrays["force_trend"][dev_indices],
            predictions["C2"]["force_trend"],
            predictions["C3"]["force_trend"],
            [0, 1, 2],
            args.bootstrap_samples,
            args.seed + 1,
        ),
        "future_recovery_mse_c3_relative_improvement_over_c2": 1.0
        - candidates["C3"]["temporal_and_controls"]["decoder_recovery"]["full"]["future_mse"]
        / candidates["C2"]["temporal_and_controls"]["decoder_recovery"]["full"]["future_mse"],
        "future_recovery_error_c2_minus_c3_ci95": bootstrap_ci(
            errors["C2"]["full"] - errors["C3"]["full"],
            args.bootstrap_samples,
            args.seed + 2,
        ),
        "same_samples": True,
        "same_probe_protocol": True,
        "same_native_and_common_shape": True,
        "formal_test_loaded": False,
    }
    output = {
        "schema": "tactile3d-unit.s4-2ds-contact-representation-utility.v1",
        "stage": "DS2",
        "selection_data": "DS-TRAIN/DS-DEV from original TRAIN source groups only",
        "historical_reconstruction_10_percent_gate": {
            "status": "FAIL_UNCHANGED",
            "used_for_ds_selection": False,
        },
        "candidates": candidates,
        "paired_comparison": comparison,
        "formal_validation_confirmation_loaded": False,
        "formal_test_model_metrics_loaded": False,
    }
    atomic_json(args.artifacts / "contact_representation_utility.json", output)
    print(json.dumps({
        "split": {key: split_manifest[key] for key in ("ds_train_groups", "ds_dev_groups", "ds_train_pairs", "ds_dev_pairs")},
        "C2": {"eligible": candidates["C2"]["contact_only_eligible"], "contact_f1": candidates["C2"]["probes"]["contact_transition"]["macro_f1"], "force_f1": candidates["C2"]["probes"]["force_trend"]["macro_f1"]},
        "C3": {"eligible": candidates["C3"]["contact_only_eligible"], "contact_f1": candidates["C3"]["probes"]["contact_transition"]["macro_f1"], "force_f1": candidates["C3"]["probes"]["force_trend"]["macro_f1"]},
        "formal_test_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
