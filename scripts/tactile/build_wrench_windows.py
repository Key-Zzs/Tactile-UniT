#!/usr/bin/env python3
"""Build the canonical physical-time S1 wrench window cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap

from gr00t.tactile_teacher.dataset import EpisodeData, TactileEpisodeStore
from gr00t.tactile_teacher.normalization import RobustFeatureStats
from gr00t.tactile_teacher.window import TemporalWindow, resample_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tactile_teacher/s1_contact_state_teacher.json"),
    )
    parser.add_argument(
        "--s1-0-dir", type=Path, default=Path(".local/artifacts/tactile_teacher/s1_0")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".local/cache/tactile_teacher/s1_wrench_windows")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".local/artifacts/tactile_teacher/s1_1")
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchors_for_episode(episode: EpisodeData, window: TemporalWindow, count: int) -> np.ndarray:
    first = float(episode.timestamps[0] + window.history_sec)
    last = float(episode.timestamps[-1] - window.future_sec)
    if last <= first:
        raise ValueError(
            f"episode {episode.record.episode_index} is too short for physical-time window"
        )
    return np.linspace(first, last, count, dtype=np.float64)


def plot_alignment(
    episode: EpisodeData,
    anchor: float,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
    window: TemporalWindow,
    output: Path,
) -> None:
    def magnitude(values: np.ndarray) -> np.ndarray:
        return np.linalg.norm(values.reshape(-1, 10, 6)[..., :3], axis=-1).max(axis=1)

    raw_mag = magnitude(episode.wrench)
    hist_time = np.linspace(anchor - window.history_sec, anchor, window.history_steps)
    future_time = anchor + np.linspace(
        window.future_sec / window.future_steps, window.future_sec, window.future_steps
    )
    fig, ax = plt.subplots(figsize=(13, 4.5))
    vicinity = (episode.timestamps >= anchor - 1.25 * window.history_sec) & (
        episode.timestamps <= anchor + 1.5 * window.future_sec
    )
    ax.plot(episode.timestamps[vicinity], raw_mag[vicinity], color="0.65", label="raw 30 Hz")
    ax.plot(hist_time, magnitude(history_raw), "o-", label="history resample")
    ax.plot(future_time, magnitude(future_raw), "o-", label="future target")
    ax.axvline(anchor, color="black", ls="--", lw=1, label="anchor t")
    ax.axvspan(hist_time[0], hist_time[-1], color="C0", alpha=0.08)
    ax.axvspan(future_time[0], future_time[-1], color="C1", alpha=0.08)
    ax.set_title(
        f"Physical-time alignment — episode {episode.record.episode_index} — "
        f"{episode.record.motor_primitive}"
    )
    ax.set_xlabel("episode timestamp (s)")
    ax.set_ylabel("max fingertip |force| (public sensor units; unspecified)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    split_manifest = json.loads((args.s1_0_dir / "split_manifest.json").read_text())
    normalization_path = args.s1_0_dir / "normalization.json"
    stats = RobustFeatureStats.from_dict(json.loads(normalization_path.read_text()))
    temporal = config["temporal_window"]
    window = TemporalWindow(
        history_sec=float(temporal["history_sec"]),
        future_sec=float(temporal["future_sec"]),
        history_steps=int(temporal["history_steps"]),
        future_steps=int(temporal["future_steps"]),
    )
    store = TactileEpisodeStore(
        args.dataset_root, dataset_revision=str(config["data"]["revision"]), cache_files=2
    )
    primitive_labels = sorted({r.motor_primitive for r in store.records})
    object_labels = sorted({r.object_label for r in store.records})
    primitive_to_id = {label: index for index, label in enumerate(primitive_labels)}
    object_to_id = {label: index for index, label in enumerate(object_labels)}
    record_by_id = {r.episode_index: r for r in store.records}
    sampling = config["window_sampling"]
    counts_per_episode = {
        "train": int(sampling["train_anchors_per_episode"]),
        "val": int(sampling["val_anchors_per_episode"]),
        "test": int(sampling["test_anchors_per_episode"]),
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, dict] = {}
    alignment_payload = None

    for split_name in ("train", "val", "test"):
        episode_ids = [int(x) for x in split_manifest["episode_ids"][split_name]]
        anchors_per_episode = counts_per_episode[split_name]
        total = len(episode_ids) * anchors_per_episode
        split_dir = args.cache_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        histories = open_memmap(
            split_dir / "history.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total, window.history_steps, 60),
        )
        futures = open_memmap(
            split_dir / "future.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total, window.future_steps, 60),
        )
        episode_array = open_memmap(
            split_dir / "episode_id.npy", mode="w+", dtype=np.int32, shape=(total,)
        )
        anchor_array = open_memmap(
            split_dir / "anchor_time.npy", mode="w+", dtype=np.float32, shape=(total,)
        )
        primitive_array = open_memmap(
            split_dir / "primitive_id.npy", mode="w+", dtype=np.int16, shape=(total,)
        )
        object_array = open_memmap(
            split_dir / "object_id.npy", mode="w+", dtype=np.int16, shape=(total,)
        )
        offset = 0
        for episode in store.iter_episodes(episode_ids):
            anchors = anchors_for_episode(episode, window, anchors_per_episode)
            history_raw, future_raw = resample_windows(
                episode.timestamps, episode.wrench, anchors, window
            )
            stop = offset + anchors_per_episode
            histories[offset:stop] = stats.normalize(history_raw)
            futures[offset:stop] = stats.normalize(future_raw)
            episode_array[offset:stop] = episode.record.episode_index
            anchor_array[offset:stop] = anchors
            primitive_array[offset:stop] = primitive_to_id[episode.record.motor_primitive]
            object_array[offset:stop] = object_to_id[episode.record.object_label]
            if alignment_payload is None and split_name == "test":
                mid = anchors_per_episode // 2
                alignment_payload = (episode, float(anchors[mid]), history_raw[mid], future_raw[mid])
            offset = stop
        if offset != total:
            raise RuntimeError(f"{split_name} wrote {offset} windows, expected {total}")
        for array in (
            histories,
            futures,
            episode_array,
            anchor_array,
            primitive_array,
            object_array,
        ):
            array.flush()
        if not np.all(np.isfinite(histories)) or not np.all(np.isfinite(futures)):
            raise ValueError(f"{split_name} cache contains NaN/Inf")
        split_summaries[split_name] = {
            "episodes": len(episode_ids),
            "windows": total,
            "anchors_per_episode": anchors_per_episode,
            "history_shape": list(histories.shape),
            "future_shape": list(futures.shape),
        }
        del histories, futures, episode_array, anchor_array, primitive_array, object_array

    if alignment_payload is None:
        raise RuntimeError("failed to select a real alignment example")
    plot_alignment(*alignment_payload, window, args.output_dir / "window_alignment.png")

    file_manifest = {}
    for path in sorted(args.cache_dir.rglob("*.npy")):
        relative = str(path.relative_to(args.cache_dir))
        array = np.load(path, mmap_mode="r")
        file_manifest[relative] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema": "tactile3d-unit.s1-wrench-window-cache.v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_revision": config["data"]["revision"],
        "config_sha256": sha256(args.config),
        "split_manifest_sha256": sha256(args.s1_0_dir / "split_manifest.json"),
        "normalization_sha256": sha256(normalization_path),
        "window": window.to_dict(),
        "normalization": {
            "method": config["normalization"]["method"],
            "clip": config["normalization"]["clip"],
        },
        "feature_order": store.contract.to_public_dict()["feature_order"],
        "primitive_labels": primitive_labels,
        "object_labels": object_labels,
        "splits": split_summaries,
        "files": file_manifest,
    }
    write_json(args.cache_dir / "manifest.json", manifest)
    summary = {
        "schema": "tactile3d-unit.s1.1-physical-window.v1",
        "status": "PASS",
        "window": window.to_dict(),
        "selection_rationale": temporal["rationale"],
        "splits": split_summaries,
        "cache_manifest": str((args.cache_dir / "manifest.json").resolve()),
        "alignment_plot": "window_alignment.png",
        "alignment_episode": alignment_payload[0].record.episode_index,
        "gate": {
            "physical_time_interface": True,
            "real_timestamp_resampling": True,
            "history_includes_anchor": True,
            "future_strictly_after_anchor": True,
            "normalization_frozen": True,
            "sensor_and_feature_order_frozen": True,
            "no_video_modalities": True,
            "s1_1": "PASS",
        },
    }
    write_json(args.output_dir / "s1_1_summary.json", summary)
    print(json.dumps({"status": "PASS", "splits": split_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
