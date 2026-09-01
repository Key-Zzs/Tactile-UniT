#!/usr/bin/env python3
"""Audit S4.2 Contact-State source/latent dimension on train and validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.intrinsic_dimension import (  # noqa: E402
    apply_standardizer,
    dimension_metrics,
    fit_active_cop_means,
    fit_standardizer,
    mask_inactive_cop,
    sampled_pairwise_diversity,
    temporal_summary,
)
from gr00t.simulation.s4_2_dataset import canonical_json_sha256, sha256_file  # noqa: E402
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402

PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
CHECKPOINT = ROOT / ".local/experiments/simulation/s4_2/contact_teacher/selected.pt"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r/intrinsic_dimension"
CACHE = ROOT / ".local/cache/simulation/s4_2r/contact_state"
EXPECTED_CHECKPOINT_SHA256 = "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19"
SPLITS = ("train", "validation")
REGIMES = {
    0: "free_to_free",
    1: "free_to_contact",
    2: "contact_to_free",
    3: "contact_to_contact",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def encode(
    model: torch.nn.Module,
    history: np.ndarray,
    normalizer: TactileNormalization,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    normalized = normalizer.transform(history).astype(np.float32)
    rows = []
    model = model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            rows.append(model.encode(batch).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def metric_or_skip(values: np.ndarray, *, twonn: bool = False) -> dict[str, Any]:
    if len(values) < 2:
        return {"status": "INSUFFICIENT_SAMPLES", "samples": int(len(values))}
    return dimension_metrics(values, twonn=twonn, include_spectrum=False)


def strata(
    raw: dict[str, np.ndarray], dynamic_threshold: float
) -> dict[str, np.ndarray]:
    tactile = raw["current_history"][:, -1].reshape(-1, 5, 6)
    active_count = (tactile[:, :, 0] > 0.5).sum(axis=1)
    result = {
        "overall": np.ones(len(tactile), dtype=bool),
        "static": raw["force_delta_abs"] <= dynamic_threshold,
        "dynamic_q70": raw["force_delta_abs"] > dynamic_threshold,
        "boundary": np.isin(raw["contact_transition"], [1, 2]),
    }
    for code, name in REGIMES.items():
        result[name] = raw["contact_transition"] == code
    for task in sorted(np.unique(raw["task"])):
        result[f"task:{task}"] = raw["task"] == task
    for count in range(6):
        result[f"active_regions:{count}"] = active_count == count
    return result


def stratified_metrics(
    raw: dict[str, np.ndarray],
    source_frame: np.ndarray,
    source_summary: np.ndarray,
    latent: np.ndarray,
    dynamic_threshold: float,
) -> dict[str, Any]:
    output = {}
    for name, mask in strata(raw, dynamic_threshold).items():
        output[name] = {
            "samples": int(mask.sum()),
            "active_raw_frame": metric_or_skip(source_frame[mask]),
            "history_temporal_summary": metric_or_skip(source_summary[mask]),
            "latent": metric_or_skip(latent[mask]),
        }
    return output


def representations(
    raw: dict[str, np.ndarray], active_cop_means: np.ndarray
) -> dict[str, np.ndarray]:
    history = raw["current_history"].astype(np.float64)
    masked_history = mask_inactive_cop(history, active_cop_means)
    target = raw["teacher_future"].astype(np.float64)
    masked_target = mask_inactive_cop(target, active_cop_means)
    return {
        "raw_current_frame": history[:, -1],
        "masked_current_frame": masked_history[:, -1],
        "raw_history_flattened": history.reshape(len(history), -1),
        "masked_history_flattened": masked_history.reshape(len(history), -1),
        "raw_history_temporal_summary": temporal_summary(history),
        "masked_history_temporal_summary": temporal_summary(masked_history),
        "raw_first_difference": np.diff(history, axis=1).reshape(len(history), -1),
        "masked_first_difference": np.diff(masked_history, axis=1).reshape(len(history), -1),
        "raw_future_target": target.reshape(len(target), -1),
        "masked_future_target": masked_target.reshape(len(target), -1),
    }


def plot_spectrum(source: dict[str, Any], latent: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, value in (("active raw frame", source), ("Contact-State latent", latent)):
        ratio = np.asarray(value["explained_variance_ratio"], dtype=float)
        ax.semilogy(np.arange(1, len(ratio) + 1), np.maximum(ratio, 1e-12), label=label)
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio")
    ax.set_title("S4.2-R source and latent PCA spectra (validation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_single_spectrum(value: dict[str, Any], title: str, output: Path) -> None:
    ratio = np.asarray(value["explained_variance_ratio"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(ratio) + 1), np.maximum(ratio, 1e-12))
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio", title=title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_rank_group(
    values: dict[str, Any], prefixes: list[str], title: str, output: Path
) -> None:
    labels = []
    source = []
    latent = []
    for name, row in values.items():
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if "effective_rank" not in row["latent"]:
            continue
        labels.append(name.split(":", 1)[-1])
        source.append(row["active_raw_frame"]["effective_rank"])
        latent.append(row["latent"]["effective_rank"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.15), 5))
    ax.bar(x - 0.2, source, 0.4, label="active raw frame")
    ax.bar(x + 0.2, latent, 0.4, label="latent")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set(ylabel="Effective rank", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, default=PAIR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"selected checkpoint identity mismatch: {checkpoint_sha}")
    raw = {split: load_npz(args.pair_root / f"{split}.npz") for split in SPLITS}
    active_cop_means = fit_active_cop_means(raw["train"]["current_history"])
    represented = {
        split: representations(raw[split], active_cop_means) for split in SPLITS
    }
    train_flat_mean, train_flat_std = fit_standardizer(
        represented["train"]["raw_history_flattened"]
    )
    train_masked_mean, train_masked_std = fit_standardizer(
        represented["train"]["masked_history_flattened"]
    )
    for split in SPLITS:
        represented[split]["train_standardized_history_flattened"] = apply_standardizer(
            represented[split]["raw_history_flattened"], train_flat_mean, train_flat_std
        )
        represented[split]["train_standardized_masked_history_flattened"] = apply_standardizer(
            represented[split]["masked_history_flattened"], train_masked_mean, train_masked_std
        )

    model, checkpoint = load_teacher_checkpoint(args.checkpoint, "cpu")
    normalizer = TactileNormalization.from_json(checkpoint["normalization"])
    device = torch.device(args.device)
    latent = {
        split: encode(
            model,
            raw[split]["current_history"],
            normalizer,
            device,
            args.batch_size,
        )
        for split in SPLITS
    }
    args.cache.mkdir(parents=True, exist_ok=True)
    latent_cache = {}
    for split in SPLITS:
        destination = args.cache / f"{split}.npz"
        np.savez_compressed(
            destination,
            pair_id=raw[split]["pair_id"],
            episode_id=raw[split]["episode_id"],
            task=raw[split]["task"],
            source_trajectory_id=raw[split]["source_trajectory_id"],
            h_current=latent[split],
            contact_transition=raw[split]["contact_transition"],
            current_total_force=raw[split]["current_total_force"],
            future_total_force=raw[split]["future_total_force"],
            force_delta_abs=raw[split]["force_delta_abs"],
        )
        latent_cache[split] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
            "shape": list(latent[split].shape),
        }

    source_metrics: dict[str, Any] = {}
    latent_metrics: dict[str, Any] = {}
    for split in SPLITS:
        current_active = (
            raw[split]["current_history"][:, -1].reshape(-1, 5, 6)[:, :, 0] > 0.5
        ).any(axis=1)
        source_metrics[split] = {
            name: dimension_metrics(
                value,
                twonn=(
                    name.endswith("current_frame")
                    or name.endswith("history_flattened")
                    or name.endswith("history_temporal_summary")
                    or name.endswith("first_difference")
                    or name.endswith("future_target")
                ),
            )
            for name, value in represented[split].items()
        }
        source_metrics[split]["active_contact_only"] = {
            "masked_current_frame": dimension_metrics(
                represented[split]["masked_current_frame"][current_active], twonn=True
            ),
            "masked_history_flattened": dimension_metrics(
                represented[split]["masked_history_flattened"][current_active], twonn=True
            ),
            "masked_history_temporal_summary": dimension_metrics(
                represented[split]["masked_history_temporal_summary"][current_active], twonn=True
            ),
        }
        latent_metrics[split] = dimension_metrics(latent[split], twonn=True)

    dynamic_threshold = float(np.quantile(raw["train"]["force_delta_abs"], 0.70))
    stratified = {
        split: stratified_metrics(
            raw[split],
            represented[split]["masked_current_frame"],
            represented[split]["masked_history_temporal_summary"],
            latent[split],
            dynamic_threshold,
        )
        for split in SPLITS
    }
    train_active = source_metrics["train"]["active_contact_only"]["masked_current_frame"]
    train_summary = source_metrics["train"]["masked_history_temporal_summary"]
    d_src_terms = {
        "active_raw_effective_rank": train_active["effective_rank"],
        "active_raw_participation_ratio": train_active["participation_ratio"],
        "history_summary_effective_rank": train_summary["effective_rank"],
        "history_summary_participation_ratio": train_summary["participation_ratio"],
    }
    d_src = float(np.median(list(d_src_terms.values())))
    validation_latent = latent_metrics["validation"]
    validation_active = source_metrics["validation"]["active_contact_only"][
        "masked_current_frame"
    ]
    validation_summary = source_metrics["validation"]["masked_history_temporal_summary"]
    diagnostic_ratios = {
        "latent_effective_rank_over_active_frame_raw_effective_rank": validation_latent[
            "effective_rank"
        ]
        / validation_active["effective_rank"],
        "latent_effective_rank_over_history_summary_effective_rank": validation_latent[
            "effective_rank"
        ]
        / validation_summary["effective_rank"],
        "latent_participation_ratio_over_history_summary_participation_ratio": validation_latent[
            "participation_ratio"
        ]
        / validation_summary["participation_ratio"],
        "latent_d95_over_source_summary_d95": validation_latent["d95"]
        / validation_summary["d95"],
    }
    source_output = {
        "schema": "tactile3d-unit.s4-2r-source-intrinsic-dimension.v1",
        "splits_loaded": list(SPLITS),
        "test_loaded": False,
        "training_started": False,
        "contact_active_mask": {
            "definition": "inactive CoP replaced by TRAIN active-contact CoP mean for diagnostics only",
            "training_tensor_modified": False,
            "train_active_cop_means": active_cop_means.tolist(),
        },
        "dynamic_q70_threshold_newton": dynamic_threshold,
        "metrics": source_metrics,
        "stratified": stratified,
        "source_dimension_reference": {
            "d_src": d_src,
            "definition": "median of four TRAIN-only source diversity terms",
            "terms": d_src_terms,
            "canonical_sha256": canonical_json_sha256(d_src_terms),
        },
    }
    latent_output = {
        "schema": "tactile3d-unit.s4-2r-latent-intrinsic-dimension.v1",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "candidate": checkpoint["candidate"],
        "normalization": normalizer.to_json(),
        "metrics": latent_metrics,
        "stratified": stratified,
        "diagnostic_ratios": diagnostic_ratios,
        "pairwise_diversity": sampled_pairwise_diversity(latent["validation"]),
        "cache": latent_cache,
        "test_loaded": False,
        "training_started": False,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    atomic_json(args.artifacts / "source_intrinsic_dimension.json", source_output)
    atomic_json(args.artifacts / "latent_intrinsic_dimension.json", latent_output)
    plot_spectrum(
        source_metrics["validation"]["active_contact_only"]["masked_current_frame"],
        latent_metrics["validation"],
        args.artifacts / "rank_spectrum.png",
    )
    plot_single_spectrum(
        source_metrics["validation"]["masked_current_frame"],
        "Raw tactile PCA spectrum (contact-active-masked)",
        args.artifacts / "raw_tactile_pca_spectrum.png",
    )
    plot_single_spectrum(
        latent_metrics["validation"],
        "Selected Contact-State latent PCA spectrum",
        args.artifacts / "latent_pca_spectrum.png",
    )
    plot_rank_group(
        stratified["validation"],
        ["task:"],
        "Effective rank by task (validation)",
        args.artifacts / "rank_by_task.png",
    )
    plot_rank_group(
        stratified["validation"],
        list(REGIMES.values()),
        "Effective rank by contact regime (validation)",
        args.artifacts / "rank_by_contact_regime.png",
    )
    plot_rank_group(
        stratified["validation"],
        ["active_regions:"],
        "Effective rank by active-region count (validation)",
        args.artifacts / "rank_by_active_region_count.png",
    )
    labels = ["active raw", "history summary", "latent"]
    values = [
        validation_active["effective_rank"],
        validation_summary["effective_rank"],
        validation_latent["effective_rank"],
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set(ylabel="Effective rank", title="Source versus latent intrinsic dimension")
    fig.tight_layout()
    fig.savefig(args.artifacts / "source_vs_latent_intrinsic_dimension.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "R1_COMPLETE",
                "checkpoint_sha256": checkpoint_sha,
                "d_src": d_src,
                "validation_latent_effective_rank": validation_latent["effective_rank"],
                "test_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
