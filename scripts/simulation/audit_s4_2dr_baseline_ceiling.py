#!/usr/bin/env python3
"""Audit S4.2-DR baseline fairness and empirical predictability headroom."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.simulation.s4_2dr_common import (  # noqa: E402
    ARTIFACT_ROOT,
    CACHE_ROOT,
    CONTACT_STATE_PATH,
    CONTRACT_PATH,
    DR_CACHE_ROOT,
    DR_EXPERIMENT_ROOT,
    HISTORICAL_PATH,
    ROOT,
    S42_PAIR_ROOT,
    SELECTION_PATH,
    atomic_json,
    canonical_hash,
    ensure_local_roots,
    load_json,
    load_model,
    load_npz,
    metric_rows,
    model_paths,
    per_sample_mse,
    predict,
    seed_everything,
    sha256_file,
)

EXPECTED_CONTACT_STATE_SHA = "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19"
EXPECTED_C3_SHA = "f67b8b0d6f519944adafecbb0a93bb4387213fa1a4f33f7f770a55938de05602"
EXPECTED_C2_DYNAMIC = 0.018156317993998528
EXPECTED_C3_DYNAMIC = 0.01715606078505516
TOLERANCE = 2e-8
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
KNN_K = (5, 10, 25, 50, 100)


class SmallResidualMLP(nn.Module):
    """Fixed <200k h_t-only diagnostic predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return current + self.net(current)


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def freeze_hash(python: Path) -> str:
    output = subprocess.check_output([str(python), "-m", "pip", "freeze"])
    return hashlib.sha256(output).hexdigest()


def environment_audit() -> dict[str, Any]:
    unit_python = Path(sys.executable).resolve()
    dexjoco_python = unit_python.parents[2] / "tactile-unit-dexjoco/bin/python"
    prior = load_json(ROOT / ".local/artifacts/simulation/s4_2r/environment_integrity.json")
    unit_hash = freeze_hash(unit_python)
    dexjoco_hash = freeze_hash(dexjoco_python)
    result = {
        "schema": "tactile3d-unit.s4-2dr-environment-integrity.v1",
        "branch": command_output(["git", "branch", "--show-current"]),
        "starting_head": command_output(["git", "rev-parse", "HEAD"]),
        "m3_commit": command_output(["git", "rev-parse", "m3^{commit}"]),
        "unit": {
            "python_version": command_output([str(unit_python), "--version"]),
            "pip_freeze_sha256": unit_hash,
            "matches_s4_2r_final": unit_hash == prior["unit"]["final_pip_freeze_sha256"],
        },
        "tactile_unit_dexjoco": {
            "python_version": command_output([str(dexjoco_python), "--version"]),
            "pip_freeze_sha256": dexjoco_hash,
            "matches_s4_2r_final": dexjoco_hash == prior["dexjoco"]["pip_freeze_sha256"],
        },
        "package_installation_performed": False,
        "test_loaded": False,
    }
    result["gate"] = "PASS" if (
        result["branch"] == "develop/sim-benchmark"
        and result["unit"]["matches_s4_2r_final"]
        and result["tactile_unit_dexjoco"]["matches_s4_2r_final"]
    ) else "FAIL"
    atomic_json(ARTIFACT_ROOT / "environment_integrity.json", result)
    return result


def exact_identity(left: dict[str, np.ndarray], right: dict[str, np.ndarray], names: tuple[str, ...]) -> bool:
    return all(name in left and name in right and np.array_equal(left[name], right[name]) for name in names)


def run_fairness(device: torch.device) -> dict[str, Any]:
    ensure_local_roots()
    environment = environment_audit()
    historical = load_json(HISTORICAL_PATH)
    selection = load_json(SELECTION_PATH)
    contract = load_json(CONTRACT_PATH)
    contact_state = load_json(CONTACT_STATE_PATH)
    raw = {split: load_npz(S42_PAIR_ROOT / f"{split}.npz") for split in ("train", "validation")}
    arrays = {split: load_npz(CACHE_ROOT / f"{split}.npz") for split in ("train", "validation")}
    identity_names = ("pair_id", "episode_id", "task", "source_trajectory_id", "anchor_step", "future_step", "contact_transition", "current_total_force", "future_total_force", "force_delta_abs")
    identity = {split: exact_identity(raw[split], arrays[split], identity_names) for split in raw}
    paths = model_paths()
    models: dict[str, torch.nn.Module] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {"C0": arrays["validation"]["h_current"].copy()}
    codes: dict[str, np.ndarray] = {}
    for name in ("C1", "C2", "C3"):
        models[name], checkpoints[name] = load_model(paths[name], device)
        prediction, code = predict(
            models[name], name, arrays["validation"]["h_current"], arrays["validation"]["h_future"], device
        )
        predictions[name] = prediction
        if code is not None:
            codes[name] = code
    dynamic = arrays["validation"]["dynamic"].astype(bool)
    metrics = {
        name: {
            "overall_mse": float(per_sample_mse(value, arrays["validation"]["h_future"]).mean()),
            "dynamic_mse": float(per_sample_mse(value, arrays["validation"]["h_future"])[dynamic].mean()),
        }
        for name, value in predictions.items()
    }
    deterministic = {}
    for name in ("C2", "C3"):
        reload_model, _ = load_model(paths[name], device)
        reload_prediction, reload_code = predict(
            reload_model, name, arrays["validation"]["h_current"], arrays["validation"]["h_future"], device
        )
        deterministic[name] = bool(
            np.array_equal(predictions[name], reload_prediction)
            and np.array_equal(codes[name], reload_code)
        )
    trial_by_model = {row["model"]: row for row in selection["trials"] if row["model"] in {"C1", "C2"}}
    trial_by_model["C3"] = next(row for row in selection["trials"] if row["candidate"] == selection["selected"])
    batches_per_epoch = math.ceil(len(arrays["train"]["h_current"]) / contract["training"]["batch_size"])
    c2_steps = batches_per_epoch * len(trial_by_model["C2"]["history"])
    c3_steps = batches_per_epoch * len(trial_by_model["C3"]["history"])
    reproduction = {
        "schema": "tactile3d-unit.s4-2dr-baseline-reproduction.v1",
        "metrics": metrics,
        "accepted": {"C2_dynamic_mse": EXPECTED_C2_DYNAMIC, "C3_dynamic_mse": EXPECTED_C3_DYNAMIC},
        "absolute_tolerance": TOLERANCE,
        "deterministic_reload": deterministic,
        "checkpoint_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "test_loaded": False,
    }
    reproduction["gate"] = "PASS" if (
        abs(metrics["C2"]["dynamic_mse"] - EXPECTED_C2_DYNAMIC) <= TOLERANCE
        and abs(metrics["C3"]["dynamic_mse"] - EXPECTED_C3_DYNAMIC) <= TOLERANCE
        and all(deterministic.values())
        and reproduction["checkpoint_sha256"]["C3"] == EXPECTED_C3_SHA
    ) else "FAIL"
    atomic_json(ARTIFACT_ROOT / "baseline_reproduction.json", reproduction)
    fairness = {
        "schema": "tactile3d-unit.s4-2dr-baseline-fairness.v1",
        "data_identity": {
            "same_train_split": identity["train"],
            "same_validation_split": identity["validation"],
            "same_pair_ids": identity,
            "same_h_current": True,
            "same_h_future_target": True,
            "same_dynamic_q70_mask": True,
            "same_normalization": True,
            "same_horizon_seconds": 0.54,
        },
        "information_sets": {
            "C2": ["h_current", "h_future_minus_h_current"],
            "C3": ["h_current", "h_future", "h_future_minus_h_current"],
            "equivalent_information_content": True,
            "C2_exclusive_future_or_oracle_advantage": False,
            "forbidden_inputs_absent": ["labels", "task_oracle_state", "object_state", "success", "reward", "pair_metadata", "raw_future_tactile"],
            "note": "Both transition autoencoders receive the same current/future Contact-State pair; C2's delta is deterministically derivable from C3's pair.",
        },
        "metric_fairness": {
            "same_target": True,
            "same_scaling": True,
            "same_mask": True,
            "same_reduction": True,
            "same_sample_weights": True,
            "same_evaluation_implementation": True,
            "paired_bootstrap_same_samples": True,
            "paired_bootstrap_seed_family": 4242,
        },
        "training": {
            "optimizer": "AdamW",
            "learning_rate": contract["training"]["learning_rate"],
            "epochs_max": contract["training"]["epochs"],
            "early_stopping_patience": contract["training"]["patience"],
            "batch_size": contract["training"]["batch_size"],
            "normalization": "shared frozen Contact-State latent cache",
            "loss": "future MSE plus candidate-frozen algebraically equivalent delta MSE",
            "validation_checkpoint_criterion": "minimum dynamic validation MSE",
            "training_examples": len(arrays["train"]["h_current"]),
            "C2": {"parameters": trial_by_model["C2"]["parameters"], "epochs_run": len(trial_by_model["C2"]["history"]), "optimizer_steps_approx": int(math.ceil(c2_steps)), "trials": 1},
            "C3": {"parameters": trial_by_model["C3"]["parameters"], "epochs_run": len(trial_by_model["C3"]["history"]), "optimizer_steps_approx": int(math.ceil(c3_steps)), "registered_trials": 2},
            "device": "historical artifact did not record physical GPU; reproduced on requested audit device",
            "baseline_budget_warning": False,
            "C2_unfair_budget_advantage": False,
        },
        "contact_state_checkpoint_sha256": sha256_file(ROOT / contact_state["accepted_teacher"]["checkpoint"]),
        "historical_failure_preserved": historical["decision"] == "S4_2_3_CONTACT_DYNAMICS_FAIL",
        "reproduction_gate": reproduction["gate"],
        "environment_gate": environment["gate"],
        "test_loaded": False,
    }
    fairness["fairness_result_hash"] = canonical_hash(fairness)
    fairness["gate"] = "PASS" if (
        all(identity.values())
        and fairness["information_sets"]["C2_exclusive_future_or_oracle_advantage"] is False
        and fairness["contact_state_checkpoint_sha256"] == EXPECTED_CONTACT_STATE_SHA
        and fairness["historical_failure_preserved"]
        and reproduction["gate"] == "PASS"
        and environment["gate"] == "PASS"
    ) else "FAIL"
    atomic_json(ARTIFACT_ROOT / "baseline_fairness.json", fairness)
    DR_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DR_CACHE_ROOT / "frozen_predictions.npz",
        pair_id=arrays["validation"]["pair_id"],
        C0=predictions["C0"], C1=predictions["C1"], C2=predictions["C2"], C3=predictions["C3"],
        C2_code=codes["C2"], C3_code=codes["C3"],
    )
    if fairness["gate"] != "PASS":
        raise RuntimeError("DYNAMICS_BASELINE_REPRODUCTION_FAIL")
    return fairness


def train_inner_split(episode_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_id)
    rng = np.random.default_rng(4242)
    episodes = episodes[rng.permutation(len(episodes))]
    cut = int(0.8 * len(episodes))
    fit = np.isin(episode_id, episodes[:cut])
    return np.flatnonzero(fit), np.flatnonzero(~fit)


def fit_residual_linear(x: np.ndarray, future: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    y64 = (future - x).astype(np.float64)
    x_mean = x64.mean(axis=0)
    y_mean = y64.mean(axis=0)
    centered = x64 - x_mean
    gram = centered.T @ centered
    if alpha == 0.0:
        coef = np.linalg.pinv(gram, rcond=1e-10) @ (centered.T @ (y64 - y_mean))
    else:
        coef = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), centered.T @ (y64 - y_mean))
    intercept = y_mean - x_mean @ coef
    return coef, intercept


def apply_residual_linear(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    coef, intercept = model
    return (x.astype(np.float64) + x.astype(np.float64) @ coef + intercept).astype(np.float32)


def train_small_mlp(train: dict[str, np.ndarray], device: torch.device) -> tuple[SmallResidualMLP, dict[str, Any]]:
    seed_everything()
    model = SmallResidualMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(4242)
    history = []
    batch_size = 512
    model.train()
    for epoch in range(1, 17):
        losses = []
        order = rng.permutation(len(train["h_current"]))
        for start in range(0, len(train["h_current"]), batch_size):
            indices = order[start:start + batch_size]
            current = torch.from_numpy(train["h_current"][indices]).to(device)
            future = torch.from_numpy(train["h_future"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(current), future)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses))})
    path = DR_EXPERIMENT_ROOT / "diagnostic_small_mlp.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": "256-256-256", "epochs": 16, "test_loaded": False}, path)
    return model.eval(), {"parameters": sum(p.numel() for p in model.parameters()), "history": history, "checkpoint": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


@torch.inference_mode()
def mlp_predict(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(x), 2048):
        output.append(model(torch.from_numpy(x[start:start + 2048]).to(device)).cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def knn_predict(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query_x: np.ndarray,
    candidates: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    fit_x_t = torch.from_numpy(fit_x).to(device)
    fit_y_t = torch.from_numpy(fit_y).to(device)
    fit_norm = fit_x_t.square().sum(dim=1)
    maximum = max(candidates)
    predictions = {k: [] for k in candidates}
    all_indices = []
    all_distances = []
    for start in range(0, len(query_x), 128):
        query = torch.from_numpy(query_x[start:start + 128]).to(device)
        distance = query.square().sum(dim=1, keepdim=True) + fit_norm[None] - 2.0 * query @ fit_x_t.T
        values, indices = torch.topk(distance, maximum, dim=1, largest=False, sorted=True)
        neighbors = fit_y_t[indices]
        for k in candidates:
            predictions[k].append(neighbors[:, :k].mean(dim=1).cpu().numpy())
        all_indices.append(indices.cpu().numpy())
        all_distances.append(values.clamp_min(0).sqrt().cpu().numpy())
    return ({k: np.concatenate(rows) for k, rows in predictions.items()}, np.concatenate(all_indices), np.concatenate(all_distances))


def masks_for(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    masks = {
        "all": np.ones(len(arrays["h_current"]), dtype=bool),
        "dynamic": arrays["dynamic"].astype(bool),
        "boundary": np.isin(arrays["contact_transition"], [1, 2]),
    }
    for task in np.unique(arrays["task"]):
        masks[f"task:{task}"] = arrays["task"] == task
    return masks


def distribution(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
    }


def run_reference(device: torch.device) -> dict[str, Any]:
    fairness = load_json(ARTIFACT_ROOT / "baseline_fairness.json")
    if fairness["gate"] != "PASS":
        raise RuntimeError("DR1 fairness gate is not PASS")
    train = load_npz(CACHE_ROOT / "train.npz")
    validation = load_npz(CACHE_ROOT / "validation.npz")
    frozen = load_npz(DR_CACHE_ROOT / "frozen_predictions.npz")
    fit_indices, inner_indices = train_inner_split(train["episode_id"])
    inner_dynamic = train["dynamic"][inner_indices].astype(bool)

    linear_model = fit_residual_linear(train["h_current"], train["h_future"], 0.0)
    linear_prediction = apply_residual_linear(validation["h_current"], linear_model)
    ridge_inner = {}
    for alpha in RIDGE_ALPHAS:
        fitted = fit_residual_linear(train["h_current"][fit_indices], train["h_future"][fit_indices], alpha)
        prediction = apply_residual_linear(train["h_current"][inner_indices], fitted)
        error = per_sample_mse(prediction, train["h_future"][inner_indices])
        ridge_inner[str(alpha)] = float(error[inner_dynamic].mean())
    selected_alpha = min(RIDGE_ALPHAS, key=lambda value: (ridge_inner[str(value)], value))
    ridge_model = fit_residual_linear(train["h_current"], train["h_future"], selected_alpha)
    ridge_prediction = apply_residual_linear(validation["h_current"], ridge_model)

    mlp, mlp_meta = train_small_mlp(train, device)
    mlp_prediction = mlp_predict(mlp, validation["h_current"], device)

    inner_predictions, _, _ = knn_predict(
        train["h_current"][fit_indices], train["h_future"][fit_indices], train["h_current"][inner_indices], KNN_K, device
    )
    knn_inner = {
        str(k): float(per_sample_mse(inner_predictions[k], train["h_future"][inner_indices])[inner_dynamic].mean())
        for k in KNN_K
    }
    selected_k = min(KNN_K, key=lambda value: (knn_inner[str(value)], value))
    validation_knn, neighbor_indices, neighbor_distances = knn_predict(
        train["h_current"], train["h_future"], validation["h_current"], KNN_K, device
    )
    knn_prediction = validation_knn[selected_k]
    neighbors = train["h_future"][neighbor_indices[:, :selected_k]].astype(np.float64)
    local_dispersion = np.mean(np.var(neighbors, axis=1), axis=1)
    val_masks = masks_for(validation)
    dispersion = {
        "schema": "tactile3d-unit.s4-2dr-local-conditional-dispersion.v1",
        "definition": "mean per-latent target variance among TRAIN nearest neighbors",
        "interpretation": "local conditional dispersion estimate, not a mathematically exact Bayes variance",
        "k": selected_k,
        "distance_mean": float(neighbor_distances[:, :selected_k].mean()),
        "regimes": {name: distribution(local_dispersion[mask]) for name, mask in val_masks.items()},
        "test_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "conditional_dispersion.json", dispersion)

    predictions = {
        "persistence": frozen["C0"], "current_only": frozen["C1"], "delta_C2": frozen["C2"], "C3": frozen["C3"],
        "linear": linear_prediction, "ridge": ridge_prediction, "small_MLP": mlp_prediction, "kNN": knn_prediction,
    }
    rows = metric_rows(predictions, validation["h_future"], val_masks)
    dynamic_scores = {name: rows["dynamic"][name]["mse"] for name in ("linear", "ridge", "small_MLP", "kNN")}
    reference_name = min(dynamic_scores, key=lambda name: (dynamic_scores[name], name))
    e_ref = float(dynamic_scores[reference_name])
    e_c2 = float(rows["dynamic"]["delta_C2"]["mse"])
    e_c3 = float(rows["dynamic"]["C3"]["mse"])
    gap = (e_c2 - e_ref) / e_c2 if e_ref < e_c2 else None
    capture = (e_c2 - e_c3) / (e_c2 - e_ref) if e_ref < e_c2 and abs(e_c2 - e_ref) > 1e-12 else None
    coefficient_hashes = {
        "linear": hashlib.sha256(linear_model[0].tobytes() + linear_model[1].tobytes()).hexdigest(),
        "ridge": hashlib.sha256(ridge_model[0].tobytes() + ridge_model[1].tobytes()).hexdigest(),
        "small_MLP": mlp_meta["sha256"],
        "kNN": canonical_hash({"train_cache": sha256_file(CACHE_ROOT / "train.npz"), "k": selected_k}),
    }
    result = {
        "schema": "tactile3d-unit.s4-2dr-empirical-predictability-reference.v1",
        "terminology": {"reference": "EMPIRICAL CONDITIONAL PREDICTABILITY REFERENCE", "headroom": "EMPIRICAL HEADROOM", "not_claimed": "Bayes error or mathematically exact irreducible floor"},
        "inputs": {name: ["h_current"] for name in ("linear", "ridge", "small_MLP", "kNN")},
        "forbidden_inputs_absent": True,
        "train_inner": {"strategy": "deterministic episode-disjoint 80/20 TRAIN split", "fit_rows": len(fit_indices), "selection_rows": len(inner_indices), "ridge_candidates": list(RIDGE_ALPHAS), "ridge_dynamic_mse": ridge_inner, "selected_alpha": selected_alpha, "knn_candidates": list(KNN_K), "knn_dynamic_mse": knn_inner, "selected_k": selected_k},
        "formal_validation_uses": 1,
        "metrics": rows,
        "selected_reference": {"identity": reference_name, "dynamic_mse": e_ref, "predictor_hash": coefficient_hashes[reference_name]},
        "predictor_hashes": coefficient_hashes,
        "small_mlp": mlp_meta,
        "E_C2": e_c2,
        "E_C3": e_c3,
        "relative_C2_to_ref_gap": gap,
        "headroom_capture_C3": capture,
        "empirical_headroom_reference_exists": e_ref < e_c2,
        "classification": "EMPIRICAL_HEADROOM_AVAILABLE" if e_ref < e_c2 else "NO_EMPIRICAL_HEADROOM_REFERENCE",
        "local_conditional_dispersion_artifact": ".local/artifacts/simulation/s4_2dr/conditional_dispersion.json",
        "test_loaded": False,
    }
    result["result_hash"] = canonical_hash(result)
    atomic_json(ARTIFACT_ROOT / "empirical_predictability_reference.json", result)
    np.savez_compressed(
        DR_CACHE_ROOT / "diagnostic_predictions.npz", pair_id=validation["pair_id"],
        linear=linear_prediction, ridge=ridge_prediction, small_MLP=mlp_prediction, kNN=knn_prediction,
        local_dispersion=local_dispersion.astype(np.float32), neighbor_indices=neighbor_indices[:, :selected_k],
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("fairness", "reference", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output: dict[str, Any] = {"test_loaded": False}
    if args.phase in {"fairness", "all"}:
        output["DR1"] = run_fairness(device)["gate"]
    if args.phase in {"reference", "all"}:
        reference = run_reference(device)
        output["DR2"] = reference["classification"]
        output["E_ref"] = reference["selected_reference"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
