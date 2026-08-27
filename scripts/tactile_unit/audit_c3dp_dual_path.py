#!/usr/bin/env python3
"""Pretraining dual-path decomposition and Contact-private audit (train/val only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c2r_contact_preservation import (  # noqa: E402
    canonical_contact_probe,
)
from gr00t.tactile_unit.c3dp_shared_private import (  # noqa: E402
    dual_path_numpy_audit,
    private_geometry,
    sha256_file,
)
from gr00t.tactile_unit.continuous_vac_shared_space import (  # noqa: E402
    different_episode_permutation,
    numpy_flatten_normalize,
)
from gr00t.tactile_unit.vac_latent_dataset import load_split  # noqa: E402
from scripts.tactile_unit.c3dp_runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    ensure_cache_identities,
    load_config,
    load_derived_split,
    load_frozen_shared_space,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_s2_model,
)
from scripts.tactile_unit.vac_runtime_common import (  # noqa: E402
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--s2-checkpoint",
        type=Path,
        default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt",
    )
    return parser.parse_args()


def contact_probes(train, validation, train_c3, validation_c3) -> dict[str, Any]:
    representations = {
        "native": (train.arrays["z_c"], validation.arrays["z_c"]),
        "shared_u_c": (train_c3["u_c"], validation_c3["u_c"]),
        "shared_recovered": (train_c3["z_c_shared"], validation_c3["z_c_shared"]),
        "private_residual": (train_c3["r_c_priv"], validation_c3["r_c_priv"]),
    }
    result = {}
    for metric, key, classes in (
        ("contact_transition", "contact_transition", 4),
        ("force_trend", "force_trend_class", 3),
    ):
        result[metric] = {
            name: canonical_contact_probe(
                train_x, evaluation_x, train.arrays[key], validation.arrays[key], classes
            )
            for name, (train_x, evaluation_x) in representations.items()
        }
    return result


def physics_metrics(decoder, split, representations, device, batch_size):
    result = {}
    for name, value in representations.items():
        errors = np.empty(len(split), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(split), batch_size):
                stop = min(start + batch_size, len(split))
                z_value = torch.from_numpy(np.array(value[start:stop], copy=True)).to(device)
                current = torch.from_numpy(
                    np.array(split.arrays["h_current"][start:stop], copy=True)
                ).to(device)
                future = torch.from_numpy(
                    np.array(split.arrays["h_future"][start:stop], copy=True)
                ).to(device)
                prediction = decoder(z_value, current)
                errors[start:stop] = (
                    torch.square(prediction - future).mean(1).double().cpu().numpy()
                )
        dynamic = np.asarray(split.arrays["dynamic"], dtype=bool)
        transition = np.asarray(split.arrays["contact_transition"])
        result[name] = {
            "future_mse": float(errors.mean()),
            "dynamic_mse": float(errors[dynamic].mean()),
            "free_to_contact_mse": float(errors[transition == 1].mean()),
            "contact_to_free_mse": float(errors[transition == 2].mean()),
        }
    return result


def fit_ridge(
    source: np.ndarray, target: np.ndarray, alpha: float
) -> dict[str, np.ndarray | float]:
    x = np.asarray(source, dtype=np.float64).reshape(len(source), -1)
    y = np.asarray(target, dtype=np.float64).reshape(len(target), -1)
    x_mean = x.mean(0)
    y_mean = y.mean(0)
    centered = x - x_mean
    gram = centered.T @ centered
    cross = centered.T @ (y - y_mean)
    coefficient = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), cross)
    return {"alpha": float(alpha), "x_mean": x_mean, "y_mean": y_mean, "coefficient": coefficient}


def apply_ridge(model, source: np.ndarray) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64).reshape(len(source), -1)
    prediction = (x - model["x_mean"]) @ model["coefficient"] + model["y_mean"]
    return prediction.reshape(len(source), 8, 32).astype(np.float32)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    squared = np.square(prediction - target)
    denominator = np.square(target - target.mean(axis=0, keepdims=True)).sum()
    left = numpy_flatten_normalize(prediction)
    right = numpy_flatten_normalize(target)
    return {
        "mse": float(squared.mean()),
        "r2": float(1.0 - squared.sum() / max(float(denominator), 1e-12)),
        "cosine": float(np.sum(left * right, axis=1).mean()),
    }


def ridge_diagnostic(
    name,
    train_source,
    validation_source,
    train_target,
    validation_target,
    episode,
    alphas,
    seed,
    output,
):
    candidates = []
    for alpha in alphas:
        model = fit_ridge(train_source, train_target, float(alpha))
        prediction = apply_ridge(model, validation_source)
        candidates.append(
            {
                "alpha": float(alpha),
                "metrics": regression_metrics(prediction, validation_target),
                "model": model,
            }
        )
    selected = min(candidates, key=lambda value: value["metrics"]["mse"])
    model = selected.pop("model")
    shuffled = np.random.default_rng(seed).permutation(len(validation_source))
    different = different_episode_permutation(episode, seed + 1)
    controls = {
        "mean_residual": regression_metrics(
            np.broadcast_to(model["y_mean"].reshape(1, 8, 32), validation_target.shape),
            validation_target,
        ),
        "shuffled_source": regression_metrics(
            apply_ridge(model, validation_source[shuffled]), validation_target
        ),
        "different_episode_source": regression_metrics(
            apply_ridge(model, validation_source[different]), validation_target
        ),
    }
    np.savez(
        output,
        alpha=np.asarray(model["alpha"]),
        x_mean=model["x_mean"],
        y_mean=model["y_mean"],
        coefficient=model["coefficient"],
    )
    return {
        "selection_split": "validation only",
        "test_loaded": False,
        "selected_alpha": model["alpha"],
        "metrics": selected["metrics"],
        "controls": controls,
        "candidates": [{"alpha": row["alpha"], "metrics": row["metrics"]} for row in candidates],
        "model": str(output.relative_to(ROOT)),
        "model_sha256": sha256_file(output),
    }


def classify_private(diagnostics, policy):
    best_r2 = max(value["metrics"]["r2"] for value in diagnostics.values())
    material = False
    for value in diagnostics.values():
        mean_mse = value["controls"]["mean_residual"]["mse"]
        gain = (mean_mse - value["metrics"]["mse"]) / max(mean_mse, 1e-12)
        material = material or gain >= float(policy["control_mse_improvement_min"])
    if best_r2 >= float(policy["strong_r2_min"]) and material:
        return "PRIVATE_RESIDUAL_CONTAINS_STRONG_SHARED_SIGNAL"
    if best_r2 >= float(policy["mixed_r2_min"]) and material:
        return "PRIVATE_RESIDUAL_MIXED"
    return "PRIVATE_RESIDUAL_LARGELY_PRIVATE"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = config["runtime"]
    cache_root = ROOT / runtime["cache_root"]
    artifact_root = ROOT / runtime["artifact_root"]
    experiment_root = ROOT / runtime["experiment_root"]
    device, lock_handle, gpu = resolve_device(args.device, allowed_physical=("2", "3"))
    try:
        set_seed(int(config["seed"]))
        train_c3, train_manifest = load_derived_split(cache_root, "train")
        validation_c3, validation_manifest = load_derived_split(cache_root, "validation")
        ensure_cache_identities(config, train_manifest, validation_manifest)
        c1_root = ROOT / runtime["c1_cache_root"]
        train = load_split(c1_root, "train", verify_hashes=True)
        validation = load_split(c1_root, "validation", verify_hashes=True)
        model, _, checkpoint_sha = load_frozen_shared_space(config, device)
        shared_digest_before = train_manifest["shared_state_sha256"]

        decomposition = {
            split: dual_path_numpy_audit(source.arrays["z_c"], derived["z_c_shared"])
            for split, source, derived in (
                ("train", train, train_c3),
                ("validation", validation, validation_c3),
            )
        }
        decomposition_pass = all(
            value["max_abs_error"] <= 1e-6 and value["mse"] <= 1e-12
            for value in decomposition.values()
        )
        dual_path = {
            "schema": "tactile3d-unit.vac-c3dp-dual-path-audit.v1",
            "checkpoint_sha256": checkpoint_sha,
            "test_loaded": False,
            "u_c_shape": [8, 32],
            "z_c_shared_shape": [8, 32],
            "r_c_priv_shape": [8, 32],
            "definition": "z_c = R_c(u_c) + r_c_priv",
            "arithmetic_identity": decomposition,
            "pass": decomposition_pass,
        }
        if not decomposition_pass:
            raise RuntimeError("STRUCTURAL_FAIL: dual-path arithmetic identity")
        atomic_json(artifact_root / "dual_path_audit.json", dual_path)

        probes = contact_probes(train, validation, train_c3, validation_c3)
        s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
        mean_shared = (
            np.asarray(train_c3["z_c_shared"], dtype=np.float64).mean(0).astype(np.float32)
        )
        private_control = mean_shared[None] + np.asarray(validation_c3["r_c_priv"])
        physics = physics_metrics(
            s2.decoder,
            validation,
            {
                "shared_only": validation_c3["z_c_shared"],
                "private_only_mean_shared_plus_residual": private_control,
                "full_native": validation.arrays["z_c"],
            },
            device,
            args.batch_size,
        )
        geometry = private_geometry(
            validation.arrays["z_c"],
            validation_c3["u_c"],
            validation_c3["z_c_shared"],
            validation_c3["r_c_priv"],
        )
        experiment_root.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "V->r_c_priv": ridge_diagnostic(
                "vision",
                train_c3["u_v"],
                validation_c3["u_v"],
                train_c3["r_c_priv"],
                validation_c3["r_c_priv"],
                np.asarray(validation_c3["episode_id"]),
                config["evaluation"]["ridge_alphas"],
                int(config["seed"]),
                experiment_root / "private_ridge_vision.npz",
            ),
            "A->r_c_priv": ridge_diagnostic(
                "action",
                train_c3["u_a"],
                validation_c3["u_a"],
                train_c3["r_c_priv"],
                validation_c3["r_c_priv"],
                np.asarray(validation_c3["episode_id"]),
                config["evaluation"]["ridge_alphas"],
                int(config["seed"]) + 100,
                experiment_root / "private_ridge_action.npz",
            ),
        }
        classification = classify_private(
            diagnostics, config["evaluation"]["private_classification"]
        )
        shared_digest_after = train_manifest["shared_state_sha256"]
        result = {
            "schema": "tactile3d-unit.vac-c3dp-private-residual-analysis.v1",
            "selection_split": "validation only",
            "test_loaded": False,
            "gpu": gpu,
            "geometry": geometry,
            "contact_semantics": probes,
            "physics": physics,
            "private_cross_modal_predictability": diagnostics,
            "classification_policy": config["evaluation"]["private_classification"],
            "classification": classification,
            "diagnostic_only": True,
            "shared_state_before": shared_digest_before,
            "shared_state_after": shared_digest_after,
            "shared_space_unchanged": shared_digest_before == shared_digest_after,
        }
        if not result["shared_space_unchanged"]:
            raise RuntimeError("STRUCTURAL_FAIL: shared space changed during private audit")
        atomic_json(artifact_root / "private_residual_analysis.json", result)
        print(json.dumps({"dual_path": dual_path, "private": result}, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
