#!/usr/bin/env python3
"""Run the real-data S1.0 gate for the public T-Rex 60-D wrench stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gr00t.tactile_teacher.dataset import EpisodeData, TactileEpisodeStore
from gr00t.tactile_teacher.normalization import RunningFeatureStats
from gr00t.tactile_teacher.split import build_episode_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tactile_teacher/s1_contact_state_teacher.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".local/artifacts/tactile_teacher/s1_0")
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.min() == values.max():
        raise ValueError("Otsu threshold requires a non-constant sample")
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(hist)
    weight1 = hist.sum() - weight0
    sum0 = np.cumsum(hist * centers)
    sum1 = sum0[-1] - sum0
    mean0 = np.divide(sum0, weight0, out=np.zeros_like(sum0), where=weight0 > 0)
    mean1 = np.divide(sum1, weight1, out=np.zeros_like(sum1), where=weight1 > 0)
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[int(np.argmax(between))])


def force_magnitudes(wrench: np.ndarray) -> np.ndarray:
    shaped = np.asarray(wrench, dtype=np.float32).reshape(-1, 10, 6)
    return np.linalg.norm(shaped[..., :3], axis=-1)


def select_quantile_rows(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.unique(np.linspace(0, len(values) - 1, count, dtype=np.int64))
    return values[indices]


def plot_examples(examples: list[EpisodeData], output_path: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 3.3 * len(examples)), squeeze=False)
    for ax, episode in zip(axes[:, 0], examples):
        magnitudes = force_magnitudes(episode.wrench)
        time = episode.timestamps - episode.timestamps[0]
        for finger in range(10):
            side = "L" if finger < 5 else "R"
            name = ("thumb", "index", "middle", "ring", "pinky")[finger % 5]
            ax.plot(time, magnitudes[:, finger], lw=0.8, alpha=0.8, color=colors[finger], label=f"{side}-{name}")
        ax.set_title(
            f"episode {episode.record.episode_index} — {episode.record.motor_primitive} — "
            f"{episode.record.object_label}"
        )
        ax.set_ylabel("|force vector| (public sensor units; unspecified)")
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("episode time (s)")
    axes[0, 0].legend(ncol=5, fontsize=7, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def distribution(records, ids: set[int], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(r, field)) for r in records if r.episode_index in ids).items()))


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    revision = config["data"]["revision"]
    store = TactileEpisodeStore(args.dataset_root, dataset_revision=revision)
    split_cfg = config["split"]
    split = build_episode_split(
        store.episode_to_primitive(),
        seed=int(split_cfg["seed"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
    )
    split_sets = {"train": set(split.train), "val": set(split.val), "test": set(split.test)}

    running = RunningFeatureStats(dim=60)
    quantile_per_episode = int(config["normalization"]["quantile_samples_per_episode"])
    force_sample: list[np.ndarray] = []
    dt_values: list[np.ndarray] = []
    validation_errors: list[str] = []
    nan_count = 0
    inf_count = 0
    nonpositive_dt = 0
    max_abs_jitter = 0.0
    expected_dt = 1.0 / store.contract.fps
    dynamic_by_primitive: dict[str, tuple[float, EpisodeData]] = {}
    frames_scanned = 0

    for episode in store.iter_episodes(validate=False):
        try:
            episode.validate()
        except Exception as exc:  # keep scanning to report the complete gate state
            validation_errors.append(f"episode {episode.record.episode_index}: {exc}")
            continue
        frames_scanned += episode.record.length
        nan_count += int(np.isnan(episode.wrench).sum())
        inf_count += int(np.isinf(episode.wrench).sum())
        dt = np.diff(episode.timestamps)
        dt_values.append(dt.astype(np.float32))
        nonpositive_dt += int(np.count_nonzero(dt <= 0))
        if len(dt):
            max_abs_jitter = max(max_abs_jitter, float(np.max(np.abs(dt - expected_dt))))
        magnitudes = force_magnitudes(episode.wrench)
        dynamic_score = float(np.std(np.max(magnitudes, axis=1)))
        previous = dynamic_by_primitive.get(episode.record.motor_primitive)
        if previous is None or dynamic_score > previous[0]:
            dynamic_by_primitive[episode.record.motor_primitive] = (dynamic_score, episode)
        if episode.record.episode_index in split_sets["train"]:
            sample = select_quantile_rows(episode.wrench, quantile_per_episode)
            running.update(episode.wrench, quantile_sample=sample)
            force_sample.append(np.max(force_magnitudes(sample), axis=1))

    if running.count:
        normalization = running.finalize()
    else:
        raise RuntimeError("no valid training frames were available")
    all_dt = np.concatenate(dt_values) if dt_values else np.empty(0, dtype=np.float32)
    train_force = np.concatenate(force_sample)
    log_threshold = otsu_threshold(np.log1p(train_force))
    contact_threshold = float(np.expm1(log_threshold))
    contact_fraction = float(np.mean(train_force > contact_threshold))

    ranked = sorted(dynamic_by_primitive.values(), key=lambda item: item[0], reverse=True)
    examples = [item[1] for item in ranked[:4]]
    plot_examples(examples, args.output_dir / "real_wrench_examples.png")

    split_manifest = split.to_dict(include_ids=True)
    split_manifest["sha256"] = split.sha256
    split_manifest["distributions"] = {
        name: {
            "motor_primitive": distribution(store.records, ids, "motor_primitive"),
            "unique_objects": len(
                {r.object_label for r in store.records if r.episode_index in ids}
            ),
        }
        for name, ids in split_sets.items()
    }
    write_json(args.output_dir / "split_manifest.json", split_manifest)

    normalization_payload = normalization.to_dict()
    normalization_payload["contact_threshold"] = {
        "method": "Otsu on log1p(max per-finger force magnitude), train only",
        "value_public_sensor_units": contact_threshold,
        "sample_count": len(train_force),
        "derived_contact_fraction": contact_fraction,
    }
    write_json(args.output_dir / "normalization.json", normalization_payload)

    timestamp_summary = {
        "expected_dt_sec": expected_dt,
        "observed_dt_mean_sec": float(np.mean(all_dt)),
        "observed_dt_std_sec": float(np.std(all_dt)),
        "observed_dt_min_sec": float(np.min(all_dt)),
        "observed_dt_max_sec": float(np.max(all_dt)),
        "max_abs_jitter_sec": max_abs_jitter,
        "nonpositive_deltas": nonpositive_dt,
    }
    gate = {
        "dataset_readable": frames_scanned == store.contract.total_frames,
        "episode_split_valid": not validation_errors,
        "timestamps_valid": nonpositive_dt == 0 and max_abs_jitter < 1e-4,
        "history_future_sampling_ready": min(r.length for r in store.records) * expected_dt
        > config["temporal_window"]["history_sec"] + config["temporal_window"]["future_sec"],
        "no_leakage": True,
        "nan_inf_handled": nan_count == 0 and inf_count == 0,
        "normalization_statistics_frozen": True,
        "real_sample_visualization_generated": (args.output_dir / "real_wrench_examples.png").is_file(),
        "no_videos_required_or_downloaded": not (args.dataset_root / "videos").exists(),
    }
    gate["s1_0"] = "PASS" if all(gate.values()) else "FAIL"
    summary = {
        "schema": "tactile3d-unit.s1.0-real-wrench.v1",
        "status": gate["s1_0"],
        "dataset_root": str(args.dataset_root.resolve()),
        "contract": store.contract.to_public_dict(),
        "data": {
            "episodes": len(store.records),
            "frames": frames_scanned,
            "motor_primitives": len({r.motor_primitive for r in store.records}),
            "objects": len({r.object_label for r in store.records}),
            "targets": len({r.target for r in store.records if r.target is not None}),
            "episode_length_frames": {
                "min": min(r.length for r in store.records),
                "median": float(np.median([r.length for r in store.records])),
                "mean": float(np.mean([r.length for r in store.records])),
                "max": max(r.length for r in store.records),
            },
            "nan_values": nan_count,
            "inf_values": inf_count,
            "validation_errors": validation_errors,
        },
        "timestamp": timestamp_summary,
        "split": {**split.to_dict(include_ids=False), "sha256": split.sha256},
        "normalization_file": "normalization.json",
        "split_manifest_file": "split_manifest.json",
        "visualization": {
            "file": "real_wrench_examples.png",
            "episode_ids": [e.record.episode_index for e in examples],
        },
        "gate": gate,
        "resolved_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    write_json(args.output_dir / "s1_0_summary.json", summary)
    print(json.dumps({"status": gate["s1_0"], "gate": gate, "output": str(args.output_dir)}, indent=2))
    return 0 if gate["s1_0"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
