#!/usr/bin/env python3
"""Read-only S4.3-PI2W diagnosis of the frozen PI2V UniT gate.

This program never constructs an optimizer and never calls ``train`` on the
loaded model.  It cold-loads the preregistered step-80000 adapter, evaluates
the frozen 4,860-row DEV split through the three official presence routes,
and writes only new PI2W artifacts/cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts/simulation"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2w"
CACHE = ROOT / ".local/cache/simulation/s4_3_pi2w"
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_3_pi2v/unit_adapter/checkpoint-step80000.pt"
DEV_PAIRS = ROOT / ".local/cache/simulation/s4_2/pairs/validation.npz"
BOOTSTRAP_SEED = 4243
BOOTSTRAP_RESAMPLES = 10_000

sys.path.insert(0, str(SCRIPT_ROOT))
from s4_3_pi2v_unit_adapter import frozen_parameter_digest, load_adapter_state_dict, sha256_file  # noqa: E402
from train_s4_3_pi2v_unit_adapter import OriginalUniTPairDataset, load_model  # noqa: E402


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class IndexedDataset(Dataset):
    def __init__(self, dataset: OriginalUniTPairDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return index, self.dataset[(index, 0)]


def route(model, vision, action, pv: int, pa: int):
    batch = vision.shape[0]
    device = vision.device
    unit = model.fusion(
        visual_tokens=vision,
        action_tokens=action,
        pv=torch.full((batch,), pv, dtype=torch.long, device=device),
        pa=torch.full((batch,), pa, dtype=torch.long, device=device),
    ).to(dtype=model.dtype)
    pre_rq = model.vq_down_resampler(unit)
    post_rq, indices, _ = model.vq(pre_rq)
    projected = model.bridge_projector(post_rq)
    prediction = model.vision_decoder(
        cond_input=model._pi2w_obs_embeddings,
        latent_motion_tokens=projected,
    )
    return unit, pre_rq, post_rq, indices, projected, prediction


def cosine_patch_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Evaluate the exact cosine definition in FP32.  The official model runs
    # its BF16 forward path correctly, but BF16 cosine can round a value just
    # above one, which makes the mathematically non-negative patch motion
    # diagnostic slightly negative and corrupts change-concentration ratios.
    return 1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 100):
        stop = min(start + 100, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def describe(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "bootstrap_95ci": bootstrap_ci(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def group_key(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def grouped_means(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        group_key(group): float(values[groups == group].mean())
        for group in np.unique(groups)
    }


def effective_rank(values: np.ndarray) -> dict[str, float]:
    centered = values.astype(np.float64) - values.mean(axis=0, dtype=np.float64)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eig = np.linalg.eigvalsh(covariance)
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()
    if total <= 0:
        return {"entropy_effective_rank": 0.0, "participation_ratio": 0.0}
    probability = eig[eig > 0] / total
    return {
        "entropy_effective_rank": float(np.exp(-(probability * np.log(probability)).sum())),
        "participation_ratio": float(total * total / np.square(eig).sum()),
    }


def code_usage(indices: np.ndarray) -> dict[str, dict[str, float | int]]:
    if indices.ndim == 2:
        indices = indices[..., None]
    result = {}
    for level in range(indices.shape[-1]):
        counts = np.bincount(indices[..., level].reshape(-1), minlength=128)
        p = counts[counts > 0] / counts.sum()
        entropy = float(-(p * np.log(p)).sum())
        result[f"level_{level + 1}"] = {
            "active_codes": int(np.count_nonzero(counts)),
            "entropy_nats": entropy,
            "normalized_entropy": entropy / math.log(128),
            "perplexity": float(math.exp(entropy)),
        }
    return result


def selected_mean(values: torch.Tensor, order: torch.Tensor, count: int, largest: bool) -> torch.Tensor:
    indices = order[:, -count:] if largest else order[:, :count]
    return torch.gather(values, 1, indices).mean(dim=1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    outputs = [
        ARTIFACTS / "dev_branch_diagnostic.json",
        ARTIFACTS / "dev_branch_paired_statistics.json",
        ARTIFACTS / "rq_localization.json",
        ARTIFACTS / "static_dominance_diagnostic.json",
        ARTIFACTS / "high_change_patch_diagnostic.json",
    ]
    if any(path.exists() for path in outputs):
        raise SystemExit("refusing to overwrite existing PI2W diagnostic artifacts")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or torch.cuda.device_count() != 1:
        raise SystemExit("PI2W diagnosis requires exactly one explicitly visible idle GPU")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != 80_000:
        raise RuntimeError("not the frozen formal step-80000 adapter checkpoint")
    device = torch.device("cuda:0")
    model, _ = load_model(device, seed=42)
    load_adapter_state_dict(model, checkpoint["adapter_state_dict"])
    model.eval()
    frozen_before = frozen_parameter_digest(model)

    dataset = OriginalUniTPairDataset("validation", augment=False, seed=42)
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    with np.load(DEV_PAIRS, allow_pickle=False) as source:
        groups = source["source_trajectory_id"].copy()
        pair_ids = source["pair_id"].copy()

    paths = ("D0_no_motion", "D1_vision_only", "D2_action_only", "D3_fusion")
    sample_values: dict[str, list[np.ndarray]] = {path: [] for path in paths}
    strata = ("all_patches", "top_10_percent", "top_25_percent", "bottom_50_percent")
    stratum_values = {path: {name: [] for name in strata} for path in paths}
    motion_rows: list[np.ndarray] = []
    motion_concentration_10: list[np.ndarray] = []
    motion_concentration_25: list[np.ndarray] = []
    current_future_similarity: list[np.ndarray] = []
    indices_rows: dict[str, list[np.ndarray]] = {path: [] for path in paths[1:]}
    pre_rows: dict[str, list[np.ndarray]] = {path: [] for path in paths[1:]}
    post_rows: dict[str, list[np.ndarray]] = {path: [] for path in paths[1:]}

    for batch_number, (row_indices, batch) in enumerate(loader, start=1):
        batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        obs, goal, action_inputs, _ = model.prepare_input(batch)
        vision, obs_features, future_features = model.vision_branch(obs, goal, batch_size=len(row_indices))
        action, _ = model.action_branch(
            actions=action_inputs["action"],
            state=action_inputs["state"],
            cat_ids=action_inputs["embodiment_id"],
        )
        model._pi2w_obs_embeddings = obs_features
        results = {
            "D1_vision_only": route(model, vision, action, 1, 0),
            "D2_action_only": route(model, vision, action, 0, 1),
            "D3_fusion": route(model, vision, action, 1, 1),
        }
        patch_losses = {"D0_no_motion": cosine_patch_loss(obs_features, future_features)}
        for name, result in results.items():
            patch_losses[name] = cosine_patch_loss(result[-1], future_features)
            pre_rows[name].append(result[1].float().cpu().numpy())
            post_rows[name].append(result[2].float().cpu().numpy())
            indices_rows[name].append(result[3].cpu().numpy().astype(np.int16))

        motion = patch_losses["D0_no_motion"]
        order = torch.argsort(motion, dim=1)
        total_motion = motion.sum(dim=1).clamp(min=1e-12)
        top10_count = max(1, math.ceil(motion.shape[1] * 0.10))
        top25_count = max(1, math.ceil(motion.shape[1] * 0.25))
        bottom50_count = max(1, math.floor(motion.shape[1] * 0.50))
        motion_rows.append(motion.float().cpu().numpy())
        current_future_similarity.append((1.0 - motion.mean(dim=1)).float().cpu().numpy())
        motion_concentration_10.append(
            (torch.gather(motion, 1, order[:, -top10_count:]).sum(dim=1) / total_motion).float().cpu().numpy()
        )
        motion_concentration_25.append(
            (torch.gather(motion, 1, order[:, -top25_count:]).sum(dim=1) / total_motion).float().cpu().numpy()
        )
        for name, losses in patch_losses.items():
            whole = losses.mean(dim=1)
            sample_values[name].append(whole.float().cpu().numpy())
            stratum_values[name]["all_patches"].append(whole.float().cpu().numpy())
            stratum_values[name]["top_10_percent"].append(
                selected_mean(losses, order, top10_count, True).float().cpu().numpy()
            )
            stratum_values[name]["top_25_percent"].append(
                selected_mean(losses, order, top25_count, True).float().cpu().numpy()
            )
            stratum_values[name]["bottom_50_percent"].append(
                selected_mean(losses, order, bottom50_count, False).float().cpu().numpy()
            )
        delattr(model, "_pi2w_obs_embeddings")
        if batch_number % 20 == 0 or batch_number == len(loader):
            print(f"PI2W DEV inference {batch_number}/{len(loader)}", flush=True)

    sample = {name: np.concatenate(rows) for name, rows in sample_values.items()}
    strata_arrays = {
        path: {name: np.concatenate(rows) for name, rows in by_stratum.items()}
        for path, by_stratum in stratum_values.items()
    }
    motion = np.concatenate(motion_rows)
    similarity = np.concatenate(current_future_similarity)
    concentration10 = np.concatenate(motion_concentration_10)
    concentration25 = np.concatenate(motion_concentration_25)
    pre = {name: np.concatenate(rows) for name, rows in pre_rows.items()}
    post = {name: np.concatenate(rows) for name, rows in post_rows.items()}
    codes = {name: np.concatenate(rows) for name, rows in indices_rows.items()}
    if any(len(values) != 4_860 for values in sample.values()):
        raise RuntimeError("PI2W diagnosis did not cover the exact 4,860-row DEV split")

    branch = {
        "schema": "tactile3d-unit.s4-3-pi2w-dev-branch-diagnostic.v1",
        "status": "PASS",
        "execution": "READ_ONLY_INFERENCE",
        "checkpoint": "$REPO_ROOT/.local/experiments/simulation/s4_3_pi2v/unit_adapter/checkpoint-step80000.pt",
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "rows": len(dataset),
        "source_groups": int(len(np.unique(groups))),
        "pair_id_sha256": hashlib.sha256(pair_ids.tobytes()).hexdigest(),
        "common_metric": "per patch 1-cosine(prediction,target) over dim=-1; patch mean; sample distribution",
        "paths": {},
    }
    for name in paths:
        row = describe(sample[name])
        row["per_source_group_mean"] = grouped_means(sample[name], groups)
        if name != "D0_no_motion":
            row["fraction_samples_beating_D0"] = float(np.mean(sample[name] < sample["D0_no_motion"]))
        branch["paths"][name] = row

    paired = {
        "schema": "tactile3d-unit.s4-3-pi2w-dev-branch-paired-statistics.v1",
        "status": "PASS",
        "sign_convention": "model_minus_D0; negative beats static persistence",
        "comparisons": {},
    }
    for name in paths[1:]:
        delta = sample[name] - sample["D0_no_motion"]
        paired["comparisons"][f"{name}_minus_D0"] = {
            **describe(delta),
            "fraction_negative": float(np.mean(delta < 0)),
            "per_source_group_mean": grouped_means(delta, groups),
        }

    route_rq = {}
    for name in paths[1:]:
        pre_flat = pre[name].reshape(-1, pre[name].shape[-1])
        post_flat = post[name].reshape(-1, post[name].shape[-1])
        token_cos = np.sum(pre_flat * post_flat, axis=1) / np.maximum(
            np.linalg.norm(pre_flat, axis=1) * np.linalg.norm(post_flat, axis=1), 1e-12
        )
        route_rq[name] = {
            "pre_post_cosine": describe(token_cos),
            "quantization_mse": describe(np.square(pre[name] - post[name]).mean(axis=(1, 2))),
            "pre_rq_effective_rank": effective_rank(pre_flat),
            "post_rq_effective_rank": effective_rank(post_flat),
            "code_usage": code_usage(codes[name]),
        }
    pre_va = pre["D1_vision_only"].reshape(len(dataset), -1)
    pre_aa = pre["D2_action_only"].reshape(len(dataset), -1)
    post_va = post["D1_vision_only"].reshape(len(dataset), -1)
    post_aa = post["D2_action_only"].reshape(len(dataset), -1)
    pre_relation = np.sum(pre_va * pre_aa, axis=1) / np.maximum(
        np.linalg.norm(pre_va, axis=1) * np.linalg.norm(pre_aa, axis=1), 1e-12
    )
    post_relation = np.sum(post_va * post_aa, axis=1) / np.maximum(
        np.linalg.norm(post_va, axis=1) * np.linalg.norm(post_aa, axis=1), 1e-12
    )
    rq = {
        "schema": "tactile3d-unit.s4-3-pi2w-rq-localization.v1",
        "status": "PASS",
        "routes": route_rq,
        "paired_relation_preservation": {
            "pre_rq_vision_action_cosine": describe(pre_relation),
            "post_rq_vision_action_cosine": describe(post_relation),
            "pearson_pre_vs_post": float(np.corrcoef(pre_relation, post_relation)[0, 1]),
            "post_minus_pre": describe(post_relation - pre_relation),
        },
        "decoder_interface": {
            "accepts": "bridge_projector(post_RQ_tokens)",
            "pre_RQ_decode_attempted": False,
            "reason": "Feeding pre-RQ values through an untrained interface would be a false decoding diagnostic.",
        },
    }

    high_change = {
        "schema": "tactile3d-unit.s4-3-pi2w-high-change-patch-diagnostic.v1",
        "status": "PASS",
        "diagnostic_only_not_a_replacement_gate": True,
        "selection": "within each sample, rank patches by D0 current-to-future cosine change",
        "results": {},
    }
    for stratum in strata:
        high_change["results"][stratum] = {}
        for name in paths:
            values = strata_arrays[name][stratum]
            row = describe(values)
            if name != "D0_no_motion":
                improvement = strata_arrays["D0_no_motion"][stratum] - values
                row["motion_explanation_improvement_D0_minus_model"] = describe(improvement)
                row["fraction_samples_beating_D0"] = float(np.mean(improvement > 0))
            high_change["results"][stratum][name] = row

    static = {
        "schema": "tactile3d-unit.s4-3-pi2w-static-dominance-diagnostic.v1",
        "status": "PASS",
        "rows": len(dataset),
        "patches_per_row": int(motion.shape[1]),
        "current_to_future_cosine_similarity": {
            **describe(similarity),
            "per_source_group_mean": grouped_means(similarity, groups),
            "quantiles": {
                str(q): float(np.quantile(similarity, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
            },
        },
        "patch_motion_magnitude": {
            **describe(motion.reshape(-1)),
            "quantiles": {
                str(q): float(np.quantile(motion, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
            },
        },
        "change_concentration": {
            "top_10_percent_fraction_of_cosine_change": describe(concentration10),
            "top_25_percent_fraction_of_cosine_change": describe(concentration25),
        },
        "motion_explanation_improvement": {
            name: describe(sample["D0_no_motion"] - sample[name]) for name in paths[1:]
        },
    }

    frozen_after = frozen_parameter_digest(model)
    for payload in (branch, paired, rq, static, high_change):
        payload["frozen_official_parameter_digest_before"] = frozen_before
        payload["frozen_official_parameter_digest_after"] = frozen_after
        payload["frozen_official_parameters_unchanged"] = frozen_before == frozen_after
    if frozen_before != frozen_after:
        raise RuntimeError("official frozen UniT parameters changed during PI2W inference")

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE / "dev_diagnostic_arrays.npz",
        groups=groups,
        current_future_similarity=similarity,
        patch_motion=motion,
        **{f"loss_{name}": values for name, values in sample.items()},
        **{
            f"loss_{path}_{stratum}": values
            for path, by_stratum in strata_arrays.items()
            for stratum, values in by_stratum.items()
        },
        **{f"pre_{name}": values for name, values in pre.items()},
        **{f"post_{name}": values for name, values in post.items()},
    )
    atomic_json(ARTIFACTS / "dev_branch_diagnostic.json", branch)
    atomic_json(ARTIFACTS / "dev_branch_paired_statistics.json", paired)
    atomic_json(ARTIFACTS / "rq_localization.json", rq)
    atomic_json(ARTIFACTS / "static_dominance_diagnostic.json", static)
    atomic_json(ARTIFACTS / "high_change_patch_diagnostic.json", high_change)
    print(json.dumps({"status": "PASS", "rows": len(dataset), "paths": branch["paths"]}, sort_keys=True))


if __name__ == "__main__":
    main()
