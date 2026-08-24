#!/usr/bin/env python3
"""Run the canonical S1.4 representation benchmark on frozen checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from gr00t.tactile_teacher.cache import load_split_arrays
from gr00t.tactile_teacher.evaluation import (
    classification_metrics,
    collapse_diagnostics,
    corrupt_history,
    regression_metrics,
    temporal_variant,
)
from gr00t.tactile_teacher.models import PredictiveContactTeacher, build_baseline
from gr00t.tactile_teacher.normalization import RobustFeatureStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".local/cache/tactile_teacher/s1_wrench_windows"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path(".local/artifacts/tactile_teacher/s1_0/normalization.json"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(".local/experiments/tactile_teacher/s1_baselines"),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path(".local/experiments/tactile_teacher/s1_teacher/best.pt"),
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=Path(".local/cache/tactile_teacher/s1_latents"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/tactile_teacher/s1_4"),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_models(args: argparse.Namespace, device: torch.device) -> dict[str, torch.nn.Module]:
    models = {}
    for name in ("B0", "B1", "B2", "B3"):
        checkpoint = torch.load(
            args.baseline_dir / name / "best.pt", map_location=device, weights_only=False
        )
        model = build_baseline(name, latent_dim=int(checkpoint["latent_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        models[name] = model.to(device).eval()
    checkpoint = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher = PredictiveContactTeacher(
        latent_dim=int(checkpoint["latent_dim"]), channels=int(checkpoint["channels"])
    )
    teacher.load_state_dict(checkpoint["state_dict"])
    models["teacher"] = teacher.to(device).eval()
    return models


@torch.inference_mode()
def extract_model(
    model: torch.nn.Module,
    history: np.ndarray,
    output_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = output_dir / f"{split}_latent.npy"
    prediction_path = output_dir / f"{split}_future.npy"
    latent_dim = int(model.latent_dim if hasattr(model, "latent_dim") else model.config.latent_dim)
    latents = open_memmap(
        latent_path, mode="w+", dtype=np.float32, shape=(len(history), latent_dim)
    )
    predictions = None
    if split == "test":
        predictions = open_memmap(
            prediction_path, mode="w+", dtype=np.float32, shape=(len(history), 8, 60)
        )
    for start in range(0, len(history), batch_size):
        stop = min(start + batch_size, len(history))
        batch = torch.from_numpy(np.array(history[start:stop], copy=True)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch)
        latents[start:stop] = output["latent"].float().cpu().numpy()
        if predictions is not None:
            predictions[start:stop] = output["future"].float().cpu().numpy()
    latents.flush()
    if predictions is not None:
        predictions.flush()
    return np.load(latent_path, mmap_mode="r"), (
        np.load(prediction_path, mmap_mode="r") if predictions is not None else None
    )


def raw_force_magnitudes(
    normalized: np.ndarray, stats: RobustFeatureStats
) -> np.ndarray:
    raw = stats.denormalize(normalized)
    shaped = raw.reshape(*raw.shape[:-1], 10, 6)
    return np.linalg.norm(shaped[..., :3], axis=-1)


def prepare_labels(
    arrays: dict[str, np.ndarray],
    stats: RobustFeatureStats,
    contact_threshold: float,
    deadband: float | None,
) -> tuple[dict[str, np.ndarray], float]:
    current_fingers = raw_force_magnitudes(np.asarray(arrays["history"][:, -1]), stats)
    future_fingers = raw_force_magnitudes(np.asarray(arrays["future"][:, -1]), stats)
    current = current_fingers.max(axis=-1)
    future = future_fingers.max(axis=-1)
    trend = future - current
    if deadband is None:
        deadband = float(np.quantile(np.abs(trend), 0.33))
    trend_class = np.ones(len(trend), dtype=np.int64)
    trend_class[trend < -deadband] = 0
    trend_class[trend > deadband] = 2
    labels = {
        "contact": (current > contact_threshold).astype(np.int64),
        "force_magnitude": current.astype(np.float32),
        "force_trend": trend.astype(np.float32),
        "force_trend_class": trend_class,
        "finger_contact": (current_fingers > contact_threshold).astype(np.float32),
        "primitive": np.asarray(arrays["primitive_id"], dtype=np.int64),
        "object": np.asarray(arrays["object_id"], dtype=np.int64),
    }
    return labels, deadband


def future_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    history: np.ndarray,
    stats: RobustFeatureStats,
    deadband: float,
) -> dict:
    current = raw_force_magnitudes(np.asarray(history[:, -1]), stats).max(axis=-1)
    target_end = raw_force_magnitudes(np.asarray(target[:, -1]), stats).max(axis=-1)
    prediction_end = raw_force_magnitudes(np.asarray(prediction[:, -1]), stats).max(axis=-1)
    trend = target_end - current
    predicted_trend = prediction_end - current
    dynamic = np.abs(trend) > deadband
    true_direction = np.sign(trend[dynamic])
    predicted_direction = np.sign(predicted_trend[dynamic])
    return {
        "all": regression_metrics(target, prediction),
        "dynamic": regression_metrics(target[dynamic], prediction[dynamic]),
        "force_trend_mae": float(np.mean(np.abs(predicted_trend - trend))),
        "dynamic_direction_accuracy": float(np.mean(true_direction == predicted_direction)),
        "dynamic_windows": int(dynamic.sum()),
    }


def train_future_mean(future: np.ndarray, batch_size: int) -> np.ndarray:
    total = np.zeros(future.shape[1:], dtype=np.float64)
    for start in range(0, len(future), batch_size):
        total += np.asarray(future[start : start + batch_size], dtype=np.float64).sum(0)
    return (total / len(future)).astype(np.float32)


TargetFunction = Callable[[int, int, torch.device], torch.Tensor]


def fit_ridge(
    latent: np.ndarray,
    target_fn: TargetFunction,
    output_dim: int,
    ridge: float,
    batch_size: int,
    device: torch.device,
) -> dict:
    mean = np.asarray(latent).mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.asarray(latent).std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    dim = latent.shape[1] + 1
    gram = torch.zeros((dim, dim), device=device)
    cross = torch.zeros((dim, output_dim), device=device)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    for start in range(0, len(latent), batch_size):
        stop = min(start + batch_size, len(latent))
        value = torch.from_numpy(np.array(latent[start:stop], copy=True)).to(device)
        value = (value - mean_tensor) / std_tensor
        value = torch.cat([value, torch.ones((len(value), 1), device=device)], dim=1)
        target = target_fn(start, stop, device).float()
        gram.add_(value.transpose(0, 1) @ value)
        cross.add_(value.transpose(0, 1) @ target)
    penalty = torch.eye(dim, device=device) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(gram + penalty, cross)
    return {"mean": mean, "std": std, "weights": weights.cpu().numpy()}


def apply_ridge(
    probe: dict, latent: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    output = np.empty((len(latent), probe["weights"].shape[1]), dtype=np.float32)
    mean = torch.from_numpy(probe["mean"]).to(device)
    std = torch.from_numpy(probe["std"]).to(device)
    weights = torch.from_numpy(probe["weights"]).to(device)
    for start in range(0, len(latent), batch_size):
        stop = min(start + batch_size, len(latent))
        value = torch.from_numpy(np.array(latent[start:stop], copy=True)).to(device)
        value = (value - mean) / std
        value = torch.cat([value, torch.ones((len(value), 1), device=device)], dim=1)
        output[start:stop] = (value @ weights).cpu().numpy()
    return output


def continuous_target(array: np.ndarray) -> TargetFunction:
    return lambda start, stop, device: torch.from_numpy(
        np.array(array[start:stop], copy=True).reshape(stop - start, -1)
    ).to(device)


def class_target(array: np.ndarray, classes: int) -> TargetFunction:
    def target(start: int, stop: int, device: torch.device) -> torch.Tensor:
        labels = torch.from_numpy(np.array(array[start:stop], copy=True)).long().to(device)
        return F.one_hot(labels, classes).float()

    return target


def fit_all_probes(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    test_arrays: dict[str, np.ndarray],
    train_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    stats: RobustFeatureStats,
    deadband: float,
    args: argparse.Namespace,
) -> dict:
    definitions = {
        "contact": (class_target(train_labels["contact"], 2), 2),
        "force_magnitude": (continuous_target(train_labels["force_magnitude"][:, None]), 1),
        "force_trend": (class_target(train_labels["force_trend_class"], 3), 3),
        "primitive": (
            class_target(train_labels["primitive"], int(train_labels["primitive"].max()) + 1),
            int(train_labels["primitive"].max()) + 1,
        ),
        "object": (
            class_target(train_labels["object"], int(train_labels["object"].max()) + 1),
            int(train_labels["object"].max()) + 1,
        ),
        "finger_contact": (continuous_target(train_labels["finger_contact"]), 10),
        "future": (continuous_target(train_arrays["future"]), 8 * 60),
    }
    results = {}
    for name, (target_fn, output_dim) in definitions.items():
        probe = fit_ridge(
            train_latent,
            target_fn,
            output_dim,
            args.ridge,
            args.batch_size,
            torch.device(args.device),
        )
        prediction = apply_ridge(
            probe, test_latent, args.batch_size, torch.device(args.device)
        )
        if name in ("contact", "force_trend", "primitive", "object"):
            label_key = "force_trend_class" if name == "force_trend" else name
            results[name] = classification_metrics(
                test_labels[label_key], prediction.argmax(axis=1)
            )
        elif name == "finger_contact":
            results[name] = {
                "micro_accuracy": float(
                    np.mean((prediction > 0.5) == test_labels["finger_contact"])
                ),
                "label_type": "DERIVED PROBE LABEL",
            }
        elif name == "future":
            results[name] = future_metrics(
                test_arrays["future"],
                prediction.reshape(-1, 8, 60),
                test_arrays["history"],
                stats,
                deadband,
            )
        else:
            results[name] = regression_metrics(
                test_labels["force_magnitude"], prediction[:, 0]
            )
    results["contact"]["label_type"] = "DERIVED PROBE LABEL"
    results["force_trend"]["label_type"] = "DERIVED PROBE LABEL"
    results["primitive"]["label_type"] = "ACTUAL METADATA"
    results["object"]["label_type"] = "ACTUAL METADATA"
    return results


@torch.inference_mode()
def evaluate_teacher_input(
    teacher: PredictiveContactTeacher,
    history: np.ndarray,
    target: np.ndarray,
    stats: RobustFeatureStats,
    deadband: float,
    args: argparse.Namespace,
    *,
    variant: str | None = None,
    corruption: str | None = None,
    severity: float = 0.0,
    clean_latent: np.ndarray | None = None,
) -> dict:
    predictions = np.empty_like(target)
    similarities = []
    device = torch.device(args.device)
    for batch_index, start in enumerate(range(0, len(history), args.batch_size)):
        stop = min(start + args.batch_size, len(history))
        value = torch.from_numpy(np.array(history[start:stop], copy=True)).to(device)
        if variant:
            value = temporal_variant(value, variant, seed=args.seed + batch_index)
        if corruption:
            value = corrupt_history(
                value, corruption, severity, seed=args.seed + batch_index
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            output = teacher(value)
        predictions[start:stop] = output["future"].float().cpu().numpy()
        if clean_latent is not None:
            clean = torch.from_numpy(np.array(clean_latent[start:stop], copy=True)).to(device)
            similarities.append(F.cosine_similarity(output["latent"].float(), clean).cpu().numpy())
    result = future_metrics(target, predictions, history, stats, deadband)
    if similarities:
        result["latent_cosine_to_clean"] = float(np.concatenate(similarities).mean())
    return result


def make_plots(
    output_dir: Path,
    direct: dict,
    temporal: dict,
    robustness: dict,
    collapse: dict,
    history: np.ndarray,
    future: np.ndarray,
    teacher_prediction: np.ndarray,
    stats: RobustFeatureStats,
    deadband: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(direct)
    values = [direct[name]["all"]["mse"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values)
    ax.set_ylabel("test future MSE (normalized wrench)")
    ax.set_title("S1.2/S1.3 direct future prediction")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png", dpi=180)
    plt.close(fig)

    names = list(temporal)
    values = [temporal[name]["dynamic"]["mse"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values)
    ax.set_ylabel("dynamic-window future MSE")
    ax.set_title("Temporal ablation — lower is better")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_ablation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for ax, (name, levels) in zip(axes.flat, robustness.items()):
        labels = list(levels)
        values = [levels[level]["all"]["mse"] for level in labels]
        ax.plot(labels, values, "o-")
        ax.set_title(name)
        ax.grid(alpha=0.2)
    fig.supylabel("future MSE")
    fig.suptitle("Teacher robustness")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_curves.png", dpi=180)
    plt.close(fig)

    eigenvalues = np.asarray(collapse["eigenvalues"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(eigenvalues / eigenvalues.sum())
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance fraction")
    ax.set_title("Teacher latent covariance spectrum")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_spectrum.png", dpi=180)
    plt.close(fig)

    current = raw_force_magnitudes(np.asarray(history[:, -1]), stats).max(axis=-1)
    future_end = raw_force_magnitudes(np.asarray(future[:, -1]), stats).max(axis=-1)
    absolute_change = np.abs(future_end - current)
    # A target-only deterministic choice: the test window nearest the 90th
    # percentile change. This avoids both cherry-picking prediction quality and
    # displaying a nearly static example from the balanced trend deadband.
    display_change = np.quantile(absolute_change, 0.9)
    index = int(np.argmin(np.abs(absolute_change - display_change)))
    history_mag = raw_force_magnitudes(np.asarray(history[index]), stats).max(axis=-1)
    future_mag = raw_force_magnitudes(np.asarray(future[index]), stats).max(axis=-1)
    prediction_mag = raw_force_magnitudes(teacher_prediction[index], stats).max(axis=-1)
    history_time = np.linspace(-16 / 30, 0, 16)
    future_time = np.linspace(1 / 30, 8 / 30, 8)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history_time, history_mag, "o-", label="observed history")
    ax.plot(future_time, future_mag, "o-", label="real future")
    ax.plot(future_time, prediction_mag, "o--", label="teacher prediction")
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("time relative to anchor (s)")
    ax.set_ylabel("max fingertip force magnitude (public units)")
    ax.set_title("Real dynamic test window: future trend prediction")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "future_prediction.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    models = load_models(args, device)
    train_arrays = load_split_arrays(args.cache_dir, "train")
    test_arrays = load_split_arrays(args.cache_dir, "test")
    normalization_payload = json.loads(args.normalization.read_text())
    stats = RobustFeatureStats.from_dict(normalization_payload)
    contact_threshold = float(
        normalization_payload["contact_threshold"]["value_public_sensor_units"]
    )
    train_labels, deadband = prepare_labels(
        train_arrays, stats, contact_threshold, deadband=None
    )
    test_labels, _ = prepare_labels(test_arrays, stats, contact_threshold, deadband=deadband)
    args.latent_cache.mkdir(parents=True, exist_ok=True)
    direct = {}
    probes = {}
    latent_paths = {}
    predictions = {}
    for name, model in models.items():
        model_dir = args.latent_cache / name
        train_latent, _ = extract_model(
            model, train_arrays["history"], model_dir, "train", args.batch_size, device
        )
        test_latent, test_prediction = extract_model(
            model, test_arrays["history"], model_dir, "test", args.batch_size, device
        )
        direct[name] = future_metrics(
            test_arrays["future"],
            test_prediction,
            test_arrays["history"],
            stats,
            deadband,
        )
        probes[name] = fit_all_probes(
            train_latent,
            test_latent,
            train_arrays,
            test_arrays,
            train_labels,
            test_labels,
            stats,
            deadband,
            args,
        )
        latent_paths[name] = test_latent
        predictions[name] = test_prediction
        print(json.dumps({"completed_model": name, "direct": direct[name]}), flush=True)

    future_mean = train_future_mean(train_arrays["future"], args.batch_size)
    persistence = np.repeat(test_arrays["history"][:, -1:, :], 8, axis=1)
    mean_prediction = np.broadcast_to(future_mean, test_arrays["future"].shape)
    trivial = {
        "persistence": future_metrics(
            test_arrays["future"], persistence, test_arrays["history"], stats, deadband
        ),
        "mean": future_metrics(
            test_arrays["future"], mean_prediction, test_arrays["history"], stats, deadband
        ),
    }

    teacher = models["teacher"]
    temporal = {}
    for variant in ("full_history", "last_frame", "shuffled_history", "reversed_history"):
        temporal[variant] = evaluate_teacher_input(
            teacher,
            test_arrays["history"],
            test_arrays["future"],
            stats,
            deadband,
            args,
            variant=variant,
        )
    clean_latent = latent_paths["teacher"]
    robustness = {}
    levels = {
        "gaussian_noise": (0.05, 0.15),
        "bias": (0.05, 0.15),
        "frame_dropout": (0.1, 0.3),
        "timestamp_jitter": (0.1, 0.25),
    }
    for corruption, (mild, strong) in levels.items():
        robustness[corruption] = {"clean": direct["teacher"]}
        for label, severity in (("mild", mild), ("strong", strong)):
            robustness[corruption][label] = evaluate_teacher_input(
                teacher,
                test_arrays["history"],
                test_arrays["future"],
                stats,
                deadband,
                args,
                corruption=corruption,
                severity=severity,
                clean_latent=clean_latent,
            )

    collapse = collapse_diagnostics(clean_latent, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_rng = np.random.default_rng(args.seed)
    sample_indices = np.sort(
        sample_rng.choice(len(clean_latent), size=min(10000, len(clean_latent)), replace=False)
    )
    np.savez(
        args.output_dir / "teacher_latent_visualization_data.npz",
        latent=np.asarray(clean_latent[sample_indices]),
        contact=test_labels["contact"][sample_indices],
        force_trend=test_labels["force_trend_class"][sample_indices],
        primitive=test_labels["primitive"][sample_indices],
        object=test_labels["object"][sample_indices],
    )
    make_plots(
        args.output_dir,
        direct,
        temporal,
        robustness,
        collapse,
        test_arrays["history"],
        test_arrays["future"],
        predictions["teacher"],
        stats,
        deadband,
    )
    summary = {
        "schema": "tactile3d-unit.s1.4-representation-benchmark.v1",
        "seed": args.seed,
        "ridge": args.ridge,
        "contact_threshold_public_sensor_units": contact_threshold,
        "force_trend_deadband_public_sensor_units": deadband,
        "label_provenance": {
            "contact": "DERIVED PROBE LABEL",
            "force_trend": "DERIVED PROBE LABEL",
            "primitive": "ACTUAL METADATA",
            "object": "ACTUAL METADATA",
        },
        "direct_future_prediction": direct,
        "frozen_latent_probes": probes,
        "trivial_future_baselines": trivial,
        "temporal_ablation": temporal,
        "robustness": robustness,
        "collapse": collapse,
    }
    write_json(args.output_dir / "s1_4_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
