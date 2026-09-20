#!/usr/bin/env python3
"""Run the frozen PI2N teacher/policy-domain probe on official policy data."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_act import FrozenS42PolicyStack  # noqa: E402


PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_policy_domain_probe.json"
CONTACT = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
VA27 = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
BRIDGE = ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt"
C3 = ROOT / ".local/experiments/simulation/s4_2r/contact_dynamics/selected.pt"
E_T = ROOT / ".local/experiments/simulation/s4_2r/contact_state/accepted.pt"


def run_root() -> Path:
    path = Path(os.environ.get("PI2N_RUN_ROOT", ROOT / ".local/experiments/simulation/s4_3_pi2n")).resolve()
    expected = Path("/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n")
    if path != expected:
        raise RuntimeError(f"PI2N run root differs from audited NAS root: {path}")
    return path


def vac_v_path() -> Path:
    return run_root() / "caches/pinch_tongs_vac_v_t27/sidecar.npz"


def cache_path() -> Path:
    return run_root() / "caches/policy_domain_representations/z_c_policy_valid.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def standardized_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_check: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_mean = x_fit.mean(axis=0, dtype=np.float64)
    x_std = x_fit.std(axis=0, dtype=np.float64)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    y_mean = y_fit.mean(axis=0, dtype=np.float64)
    y_std = y_fit.std(axis=0, dtype=np.float64)
    y_std = np.where(y_std > 1e-6, y_std, 1.0)
    x_train = (x_fit.astype(np.float64) - x_mean) / x_std
    y_train = (y_fit.astype(np.float64) - y_mean) / y_std
    gram = x_train.T @ x_train / len(x_train)
    rhs = x_train.T @ y_train / len(x_train)
    weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    prediction = ((x_check.astype(np.float64) - x_mean) / x_std) @ weights
    prediction = prediction * y_std + y_mean
    return prediction.astype(np.float32), {
        "feature_constant_dimensions": int(np.count_nonzero(x_fit.std(axis=0) <= 1e-6)),
        "label_constant_dimensions": int(np.count_nonzero(y_fit.std(axis=0) <= 1e-6)),
        "weight_frobenius_norm": float(np.linalg.norm(weights)),
        "trainable_parameters": int(weights.size),
    }


def normalized_mse(prediction: np.ndarray, target: np.ndarray, train_variance: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target) / np.maximum(train_variance, 1e-8)))


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_variance: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    groups = {
        "all": slice(0, 20),
        "future_region_occupancy": slice(0, 5),
        "region_occupancy_change": slice(5, 10),
        "future_log_force_magnitude": slice(10, 15),
        "log_force_magnitude_trend": slice(15, 20),
    }
    if int(mask.sum()) == 0:
        return {"rows": 0, "groups": {name: None for name in groups}}
    result = {}
    for name, columns in groups.items():
        result[name] = normalized_mse(
            prediction[mask, columns], target[mask, columns], train_variance[columns]
        )
    occupancy_truth = target[mask, :5] > 0.5
    occupancy_prediction = prediction[mask, :5] > 0.5
    return {
        "rows": int(mask.sum()),
        "groups": result,
        "future_occupancy_accuracy": float(np.mean(occupancy_truth == occupancy_prediction)),
    }


def episode_nmse(
    prediction: np.ndarray,
    target: np.ndarray,
    train_variance: np.ndarray,
    episode: np.ndarray,
) -> dict[int, float]:
    return {
        int(value): normalized_mse(prediction[episode == value], target[episode == value], train_variance)
        for value in np.unique(episode)
    }


def paired_interval(left: dict[int, float], right: dict[int, float], samples: int, seed: int) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    values = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "left_minus_right_episode_mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "episodes": len(values),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def geometry(value: np.ndarray, tokenized: bool) -> dict[str, Any]:
    flat = value.reshape(len(value), -1).astype(np.float64)
    centered = flat - flat.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, 1e-12)
    nonzero = probabilities > 0
    effective_rank = float(np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))))
    result = {
        "mean_dimension_variance": float(np.mean(np.var(flat, axis=0))),
        "effective_rank": effective_rank,
        "largest_eigenvalue_fraction": float(eigenvalues[-1] / max(total, 1e-12)),
        "dimensions": flat.shape[1],
    }
    if tokenized:
        tokens = value.astype(np.float64)
        normalized = tokens / np.maximum(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-12)
        similarity = normalized @ np.swapaxes(normalized, 1, 2)
        off_diagonal = ~np.eye(tokens.shape[1], dtype=bool)
        result["mean_query_cosine_diversity"] = float(np.mean(1.0 - similarity[:, off_diagonal]))
    else:
        result["mean_query_cosine_diversity"] = None
    return result


def different_episode_indices(episode: np.ndarray) -> np.ndarray:
    groups = [np.flatnonzero(episode == value) for value in np.unique(episode)]
    output = np.empty(len(episode), dtype=np.int64)
    for index, rows in enumerate(groups):
        other = groups[(index + 1) % len(groups)]
        output[rows] = other[np.arange(len(rows)) % len(other)]
    if np.any(episode == episode[output]):
        raise RuntimeError("different-episode control contains an identity episode")
    return output


@torch.inference_mode()
def main() -> None:
    results_path = ARTIFACTS / "policy_domain_probe_results.json"
    lineage_path = ARTIFACTS / "teacher_policy_domain_audit.json"
    labels_path = ARTIFACTS / "diagnostic_labels_manifest.json"
    cache = cache_path()
    if any(path.exists() for path in (results_path, lineage_path, labels_path, cache)):
        raise SystemExit("refusing to overwrite a PI2N policy-domain diagnostic output")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_NUMERICAL_PROBE":
        raise RuntimeError("policy-domain probe protocol is not frozen")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("policy-domain diagnostic requires exactly one explicitly visible GPU")

    with np.load(CONTACT, allow_pickle=False) as source:
        index = source["index"].copy()
        episode_all = source["episode_index"].copy()
        frame_all = source["frame_index"].copy()
        tactile_all = source["tactile_sim"].copy()
        h_all = source["contact_state"].copy()
        uc_all = source["contact_shared_target"].copy()
        valid = source["physical_aux_valid"].copy()
    with np.load(VA27, allow_pickle=False) as source:
        if not np.array_equal(index, source["index"]) or not np.array_equal(valid, source["va_aux_valid"]):
            raise RuntimeError("VA27 target identity differs from Contact sidecar")
        va_all = source["va_shared_target"].copy()
    vac_path = vac_v_path()
    with np.load(vac_path, allow_pickle=False) as source:
        if not np.array_equal(index, source["index"]) or not np.array_equal(valid, source["vac_aux_valid"]):
            raise RuntimeError("VAC-V target identity differs from Contact sidecar")
        vac_all = source["vac_vision_target"].copy()
    rows = np.flatnonzero(valid)
    future_rows = rows + 27
    if not np.all(episode_all[rows] == episode_all[future_rows]) or not np.all(frame_all[future_rows] == frame_all[rows] + 27):
        raise RuntimeError("policy-domain future identity is not exact episode-local +27")

    device = torch.device("cuda:0")
    stack = FrozenS42PolicyStack().eval().requires_grad_(False).to(device)
    zc = np.empty((len(rows), 8, 32), dtype=np.float32)
    uc_recomputed = np.empty_like(zc)
    batch_size = 1024
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        current = torch.from_numpy(h_all[rows[start:stop]]).to(device)
        future = torch.from_numpy(h_all[future_rows[start:stop]]).to(device)
        native = stack.contact_C3(current, future)["code"]
        shared = stack.bridge_B3.encode("contact", native)
        zc[start:stop] = native.float().cpu().numpy()
        uc_recomputed[start:stop] = shared.float().cpu().numpy()
    uc = uc_all[rows]
    uc_max_error = float(np.max(np.abs(uc_recomputed - uc)))
    if uc_max_error > 1e-5:
        raise RuntimeError(f"recomputed u_c differs from frozen sidecar: {uc_max_error}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, row_index=rows, episode_index=episode_all[rows], z_c=zc)
    temporary.replace(cache)

    current_matrix = tactile_all[rows].reshape(-1, 5, 6)
    future_matrix = tactile_all[future_rows].reshape(-1, 5, 6)
    current_occupancy = (current_matrix[:, :, 0] > 0.5).astype(np.float32)
    future_occupancy = (future_matrix[:, :, 0] > 0.5).astype(np.float32)
    current_force = np.log1p(np.linalg.norm(current_matrix[:, :, 1:3], axis=-1)).astype(np.float32)
    future_force = np.log1p(np.linalg.norm(future_matrix[:, :, 1:3], axis=-1)).astype(np.float32)
    occupancy_change = future_occupancy - current_occupancy
    force_trend = future_force - current_force
    labels = np.concatenate((future_occupancy, occupancy_change, future_force, force_trend), axis=1)
    episode = episode_all[rows]
    fit = episode % 5 != 4
    check = ~fit
    if len(np.unique(episode[fit])) != 80 or len(np.unique(episode[check])) != 20:
        raise RuntimeError("episode fit/check split identity changed")
    train_variance = np.var(labels[fit], axis=0, dtype=np.float64).astype(np.float32)
    train_mean = labels[fit].mean(axis=0)
    force_change_score = np.sum(np.abs(force_trend), axis=1)
    future_force_score = np.sum(future_force, axis=1)
    dynamic_threshold = float(np.quantile(force_change_score[fit], 0.75))
    high_force_threshold = float(np.quantile(future_force_score[fit], 0.75))
    current_any = np.any(current_occupancy > 0.5, axis=1)
    future_any = np.any(future_occupancy > 0.5, axis=1)
    subsets = {
        "all": check,
        "boundary": check & (current_any != future_any),
        "dynamic": check & ((np.any(occupancy_change != 0, axis=1)) | (force_change_score > dynamic_threshold)),
        "high_force": check & (future_force_score > high_force_threshold),
    }
    representations = {
        "h_t_c": h_all[rows],
        "z_c": zc.reshape(len(rows), -1),
        "u_c_VAC": uc.reshape(len(rows), -1),
        "u_v_VAC": vac_all[rows].reshape(len(rows), -1),
        "u_v_VA27": va_all[rows].reshape(len(rows), -1),
    }
    predictions: dict[str, np.ndarray] = {}
    fit_details = {}
    subset_metrics = {}
    per_episode = {}
    for name, value in representations.items():
        prediction, detail = standardized_ridge(value[fit], labels[fit], value[check], float(protocol["probe"]["ridge_alpha"]))
        full_prediction = np.full_like(labels, np.nan)
        full_prediction[check] = prediction
        predictions[name] = full_prediction
        fit_details[name] = detail
        subset_metrics[name] = {
            subset: metrics(full_prediction, labels, train_variance, mask)
            for subset, mask in subsets.items()
        }
        per_episode[name] = episode_nmse(prediction, labels[check], train_variance, episode[check])
    constant = np.broadcast_to(train_mean, labels.shape).copy()
    persistence = np.concatenate((current_occupancy, np.zeros_like(occupancy_change), current_force, np.zeros_like(force_trend)), axis=1)
    mismatch_local = different_episode_indices(episode[check])
    controls = {
        "TRAIN_mean": {subset: metrics(constant, labels, train_variance, mask) for subset, mask in subsets.items()},
        "current_state_persistence": {subset: metrics(persistence, labels, train_variance, mask) for subset, mask in subsets.items()},
        "different_episode_mismatch": {
            name: normalized_mse(predictions[name][check], labels[check][mismatch_local], train_variance)
            for name in predictions
        },
    }
    bootstrap_samples = int(protocol["paired_uncertainty"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["paired_uncertainty"]["bootstrap_seed"])
    comparisons = {
        "u_c_minus_z_c": paired_interval(per_episode["u_c_VAC"], per_episode["z_c"], bootstrap_samples, bootstrap_seed),
        "u_v_VAC_minus_u_c": paired_interval(per_episode["u_v_VAC"], per_episode["u_c_VAC"], bootstrap_samples, bootstrap_seed + 1),
        "u_v_VA27_minus_u_c": paired_interval(per_episode["u_v_VA27"], per_episode["u_c_VAC"], bootstrap_samples, bootstrap_seed + 2),
        "u_v_VAC_minus_u_v_VA27": paired_interval(per_episode["u_v_VAC"], per_episode["u_v_VA27"], bootstrap_samples, bootstrap_seed + 3),
    }
    geometry_results = {
        "h_t_c": geometry(h_all[rows], tokenized=False),
        "z_c": geometry(zc, tokenized=True),
        "u_c_VAC": geometry(uc, tokenized=True),
        "u_v_VAC": geometry(vac_all[rows], tokenized=True),
        "u_v_VA27": geometry(va_all[rows], tokenized=True),
    }
    overall = {name: subset_metrics[name]["all"]["groups"]["all"] for name in representations}
    shared = comparisons["u_c_minus_z_c"]
    readout = comparisons["u_v_VAC_minus_u_c"]
    diagnoses = []
    if overall["u_c_VAC"] >= 1.1 * overall["z_c"] and shared["ci95"][0] > 0:
        diagnoses.append("SHARED_CONTACT_INFORMATION_LOSS")
    if overall["u_v_VAC"] <= 0.9 * overall["u_c_VAC"] and readout["ci95"][1] < 0:
        diagnoses.append("TARGET_READOUT_MISMATCH")
    mean_boundary = controls["TRAIN_mean"]["boundary"]["groups"]["all"]
    mean_dynamic = controls["TRAIN_mean"]["dynamic"]["groups"]["all"]
    if (
        overall["z_c"] >= controls["TRAIN_mean"]["all"]["groups"]["all"]
        and overall["u_c_VAC"] >= controls["TRAIN_mean"]["all"]["groups"]["all"]
        and subset_metrics["z_c"]["boundary"]["groups"]["all"] >= mean_boundary
        and subset_metrics["u_c_VAC"]["dynamic"]["groups"]["all"] >= mean_dynamic
    ):
        diagnoses.append("TEACHER_DOMAIN_OR_SENSOR_LIMIT")
    if not diagnoses:
        diagnoses.append("INCONCLUSIVE")

    lineage = {
        "schema": "tactile3d-unit.s4-3-pi2n-teacher-policy-domain-audit.v1",
        "status": "PASS",
        "interpretation": "training-domain diagnostic with episode-held-out checks; not unseen policy generalization",
        "checkpoint_sha256": {"E_T": sha256_file(E_T), "C3": sha256_file(C3), "B3": sha256_file(BRIDGE)},
        "target_sha256": {"u_c_VAC_sidecar": sha256_file(CONTACT), "u_v_VAC": sha256_file(vac_path), "u_v_VA27": sha256_file(VA27)},
        "recomputed_u_c_max_abs_error": uc_max_error,
        "contact_influence_class": "CONTACT_INFLUENCE_SUPPORTED_BY_TRAINING_PROVENANCE",
        "teacher_comparison_level": "PACKAGE_LEVEL_TEACHER_COMPARISON_ONLY",
        "teacher_parameters_updated": False,
        "candidate_policy_checkpoints_read": False,
        "formal_rollout_outcomes_read": False,
    }
    labels_manifest = {
        "schema": "tactile3d-unit.s4-3-pi2n-diagnostic-labels.v1",
        "status": "PASS",
        "rows": len(rows),
        "fit_rows": int(fit.sum()),
        "check_rows": int(check.sum()),
        "fit_episodes": np.unique(episode[fit]).astype(int).tolist(),
        "check_episodes": np.unique(episode[check]).astype(int).tolist(),
        "dynamic_force_threshold_train_q75": dynamic_threshold,
        "high_force_threshold_train_q75": high_force_threshold,
        "check_subset_rows": {name: int(mask.sum()) for name, mask in subsets.items()},
        "inactive_or_single_class_policy": "NA",
    }
    result = {
        "schema": "tactile3d-unit.s4-3-pi2n-policy-domain-probe-results.v1",
        "status": "PASS",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "cache": "$PI2N_RUN_ROOT/caches/policy_domain_representations/z_c_policy_valid.npz",
        "cache_sha256": sha256_file(cache),
        "fit_family_count": 1,
        "fit_details": fit_details,
        "subset_metrics": subset_metrics,
        "controls": controls,
        "paired_episode_comparisons": comparisons,
        "geometry": geometry_results,
        "diagnoses": diagnoses,
        "route_status": "PENDING_g_C_AND_P4; X_NOT_SELECTED",
        "limitations": [
            "all policy trajectories were already in the behavioral-cloning training corpus",
            "linear readout performance is not sufficient evidence of physical sufficiency",
            "target scales are compared only after TRAIN normalization",
            "P4 order-preserving readout and common-reference g_C are not part of this run"
        ],
    }
    atomic_json(lineage_path, lineage)
    atomic_json(labels_path, labels_manifest)
    atomic_json(results_path, result)
    print(json.dumps({"status": result["status"], "diagnoses": diagnoses, "overall_normalized_mse": overall, "cache_sha256": result["cache_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
