#!/usr/bin/env python3
"""Build frame-exact frozen-Teacher transition pairs for S2/M2."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from gr00t.contact_dynamics.contract import (
    TransitionPairContract,
    evenly_spaced_anchors,
    validate_episode_splits,
)
from gr00t.contact_dynamics.teacher import load_frozen_teacher, parameter_digest
from gr00t.tactile_teacher.dataset import TactileEpisodeStore
from gr00t.tactile_teacher.normalization import RobustFeatureStats


HORIZONS = (8, 16, 24)
CANONICAL_HORIZON = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/contact_dynamics/s2_contact_dynamics.json"),
    )
    parser.add_argument(
        "--s1-config",
        type=Path,
        default=Path("configs/tactile_teacher/s1_contact_state_teacher.json"),
    )
    parser.add_argument(
        "--s1-artifact-dir",
        type=Path,
        default=Path(".local/artifacts/tactile_teacher/s1_0"),
    )
    parser.add_argument(
        "--s1-evaluation",
        type=Path,
        default=Path(".local/artifacts/tactile_teacher/s1_4/s1_4_summary.json"),
    )
    parser.add_argument(
        "--s1-cache-manifest",
        type=Path,
        default=Path(".local/cache/tactile_teacher/s1_wrench_windows/manifest.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".local/cache/contact_dynamics/s2_transition_pairs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".local/artifacts/contact_dynamics/s2_1"),
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--inference-batch-size", type=int, default=2048)
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


def force_magnitudes(wrench: np.ndarray) -> np.ndarray:
    shaped = np.asarray(wrench, dtype=np.float32).reshape(*wrench.shape[:-1], 10, 6)
    return np.linalg.norm(shaped[..., :3], axis=-1).astype(np.float32)


@torch.inference_mode()
def encode_history(
    teacher: torch.nn.Module,
    history: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(history), 256), dtype=np.float32)
    for start in range(0, len(history), batch_size):
        stop = min(start + batch_size, len(history))
        value = torch.from_numpy(np.ascontiguousarray(history[start:stop])).to(
            device, non_blocking=True
        )
        result[start:stop] = teacher.encode(value).float().cpu().numpy()
    return result


def make_memmaps(split_dir: Path, total: int) -> dict[str, np.memmap]:
    split_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "current": (np.float32, (total, 256)),
        "future": (np.float32, (total, 256)),
        "episode_id": (np.int32, (total,)),
        "task_id": (np.int64, (total,)),
        "anchor_frame": (np.int32, (total,)),
        "anchor_time": (np.float64, (total,)),
        "future_anchor_time": (np.float64, (total,)),
        "primitive_id": (np.int16, (total,)),
        "object_id": (np.int16, (total,)),
        "current_force": (np.float32, (total,)),
        "future_force": (np.float32, (total,)),
        "current_finger_force": (np.float32, (total, 10)),
        "future_finger_force": (np.float32, (total, 10)),
        "contact_transition": (np.int8, (total,)),
        "force_trend_class": (np.int8, (total,)),
        "finger_change": (np.int8, (total, 10)),
        "dynamic": (np.bool_, (total,)),
    }
    return {
        name: open_memmap(split_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in definitions.items()
    }


def contact_transition_class(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    result = np.empty(len(current), dtype=np.int8)
    result[(~current) & (~future)] = 0
    result[(~current) & future] = 1
    result[current & future] = 2
    result[current & (~future)] = 3
    return result


def summarize_distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "std": float(value.std()),
        "p25": float(np.quantile(value, 0.25)),
        "median": float(np.median(value)),
        "p75": float(np.quantile(value, 0.75)),
        "p95": float(np.quantile(value, 0.95)),
        "max": float(value.max()),
    }


def main() -> int:
    args = parse_args()
    start_time = time.monotonic()
    config = json.loads(args.config.read_text())
    s1_config = json.loads(args.s1_config.read_text())
    split_path = args.s1_artifact_dir / "split_manifest.json"
    normalization_path = args.s1_artifact_dir / "normalization.json"
    split_manifest = json.loads(split_path.read_text())
    stats = RobustFeatureStats.from_dict(json.loads(normalization_path.read_text()))
    s1_evaluation = json.loads(args.s1_evaluation.read_text())
    s1_cache = json.loads(args.s1_cache_manifest.read_text())
    contact_threshold = float(s1_evaluation["contact_threshold_public_sensor_units"])
    force_deadband = float(s1_evaluation["force_trend_deadband_public_sensor_units"])
    clip = float(s1_config["normalization"]["clip"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    teacher, teacher_identity = load_frozen_teacher(args.teacher_checkpoint, device)
    teacher_digest_before = parameter_digest(teacher)
    store = TactileEpisodeStore(
        args.dataset_root,
        dataset_revision=str(s1_config["data"]["revision"]),
        cache_files=2,
    )
    primitive_labels = list(s1_cache["primitive_labels"])
    object_labels = list(s1_cache["object_labels"])
    primitive_to_id = {label: index for index, label in enumerate(primitive_labels)}
    object_to_id = {label: index for index, label in enumerate(object_labels)}
    pair_counts = config["transition"]["pairs_per_episode"]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, dict] = {}
    horizon_summaries: dict[str, dict] = {}
    first_history: np.ndarray | None = None

    for split in ("train", "val", "test"):
        episode_ids = [int(value) for value in split_manifest["episode_ids"][split]]
        count = int(pair_counts[split])
        total = len(episode_ids) * count
        arrays = make_memmaps(args.cache_dir / split, total)
        transition_magnitudes = {horizon: [] for horizon in HORIZONS}
        persistence_squared = {horizon: 0.0 for horizon in HORIZONS}
        persistence_elements = {horizon: 0 for horizon in HORIZONS}
        physical_force_delta = {horizon: [] for horizon in HORIZONS}
        actual_anchor_delta = {horizon: [] for horizon in HORIZONS}
        history_spans: list[np.ndarray] = []
        pending_current: list[np.ndarray] = []
        pending_future = {horizon: [] for horizon in HORIZONS}
        pending_start = 0
        offset = 0

        def flush_pending(stop: int) -> None:
            nonlocal pending_start, pending_current, pending_future
            if not pending_current:
                return
            current_history = np.concatenate(pending_current, axis=0)
            current_latent = encode_history(
                teacher, current_history, device, args.inference_batch_size
            )
            arrays["current"][pending_start:stop] = current_latent
            for horizon in HORIZONS:
                future_history = np.concatenate(pending_future[horizon], axis=0)
                future_latent = encode_history(
                    teacher, future_history, device, args.inference_batch_size
                )
                if horizon == CANONICAL_HORIZON:
                    arrays["future"][pending_start:stop] = future_latent
                delta = future_latent - current_latent
                transition_magnitudes[horizon].append(np.linalg.norm(delta, axis=1))
                persistence_squared[horizon] += float(np.square(delta, dtype=np.float64).sum())
                persistence_elements[horizon] += int(delta.size)
            pending_start = stop
            pending_current = []
            pending_future = {horizon: [] for horizon in HORIZONS}

        for episode in store.iter_episodes(episode_ids):
            anchors = evenly_spaced_anchors(
                episode.record.length,
                count,
                history_steps=16,
                maximum_horizon_frames=max(HORIZONS),
            )
            current_indices = anchors[:, None] + np.arange(-15, 1, dtype=np.int64)[None]
            current_raw = episode.wrench[current_indices]
            current_normalized = stats.normalize(current_raw, clip=clip)
            if first_history is None:
                first_history = current_normalized[:2].copy()
            pending_current.append(current_normalized)
            canonical_future_raw = None
            for horizon in HORIZONS:
                contract = TransitionPairContract(horizon_frames=horizon)
                for anchor in (int(anchors[0]), int(anchors[-1])):
                    contract.validate_anchor(anchor, episode.record.length)
                future_indices = (
                    anchors[:, None]
                    + horizon
                    + np.arange(-15, 1, dtype=np.int64)[None]
                )
                future_raw = episode.wrench[future_indices]
                pending_future[horizon].append(stats.normalize(future_raw, clip=clip))
                if horizon == CANONICAL_HORIZON:
                    canonical_future_raw = future_raw
                current_force_h = force_magnitudes(episode.wrench[anchors]).max(axis=1)
                future_force_h = force_magnitudes(episode.wrench[anchors + horizon]).max(axis=1)
                physical_force_delta[horizon].append(future_force_h - current_force_h)
                actual_anchor_delta[horizon].append(
                    episode.timestamps[anchors + horizon] - episode.timestamps[anchors]
                )
            if canonical_future_raw is None:
                raise AssertionError("canonical future history was not generated")
            stop = offset + count
            current_fingers = force_magnitudes(episode.wrench[anchors])
            future_fingers = force_magnitudes(episode.wrench[anchors + CANONICAL_HORIZON])
            current_force = current_fingers.max(axis=1)
            future_force = future_fingers.max(axis=1)
            current_contact = current_force > contact_threshold
            future_contact = future_force > contact_threshold
            trend = future_force - current_force
            trend_class = np.ones(count, dtype=np.int8)
            trend_class[trend < -force_deadband] = 0
            trend_class[trend > force_deadband] = 2
            current_finger_contact = current_fingers > contact_threshold
            future_finger_contact = future_fingers > contact_threshold
            arrays["episode_id"][offset:stop] = episode.record.episode_index
            arrays["task_id"][offset:stop] = episode.task_indices[anchors]
            arrays["anchor_frame"][offset:stop] = anchors
            arrays["anchor_time"][offset:stop] = episode.timestamps[anchors]
            arrays["future_anchor_time"][offset:stop] = episode.timestamps[
                anchors + CANONICAL_HORIZON
            ]
            arrays["primitive_id"][offset:stop] = primitive_to_id[
                episode.record.motor_primitive
            ]
            arrays["object_id"][offset:stop] = object_to_id[episode.record.object_label]
            arrays["current_force"][offset:stop] = current_force
            arrays["future_force"][offset:stop] = future_force
            arrays["current_finger_force"][offset:stop] = current_fingers
            arrays["future_finger_force"][offset:stop] = future_fingers
            arrays["contact_transition"][offset:stop] = contact_transition_class(
                current_contact, future_contact
            )
            arrays["force_trend_class"][offset:stop] = trend_class
            arrays["finger_change"][offset:stop] = (
                future_finger_contact.astype(np.int8)
                - current_finger_contact.astype(np.int8)
                + 1
            )
            history_spans.append(
                episode.timestamps[anchors] - episode.timestamps[anchors - 15]
            )
            offset = stop
            if offset - pending_start >= args.inference_batch_size:
                flush_pending(offset)
        flush_pending(offset)
        if offset != total:
            raise RuntimeError(f"{split} wrote {offset} pairs, expected {total}")
        for array in arrays.values():
            array.flush()
        split_horizons = {}
        for horizon in HORIZONS:
            magnitude = np.concatenate(transition_magnitudes[horizon])
            force_delta = np.concatenate(physical_force_delta[horizon])
            actual_delta = np.concatenate(actual_anchor_delta[horizon])
            contract = TransitionPairContract(horizon_frames=horizon)
            split_horizons[str(horizon)] = {
                **contract.to_dict(),
                "canonical": horizon == CANONICAL_HORIZON,
                "persistence_future_mse": persistence_squared[horizon]
                / persistence_elements[horizon],
                "transition_l2": summarize_distribution(magnitude),
                "physical_max_force_delta": summarize_distribution(force_delta),
                "actual_anchor_delta_sec": summarize_distribution(actual_delta),
            }
        horizon_summaries[split] = split_horizons
        split_summaries[split] = {
            "episodes": len(episode_ids),
            "pairs": total,
            "pairs_per_episode": count,
            "history_actual_span_sec": summarize_distribution(np.concatenate(history_spans)),
        }
        del arrays
        print(json.dumps({"split": split, "pairs": total}), flush=True)

    train_arrays = {
        name: np.load(args.cache_dir / "train" / f"{name}.npy", mmap_mode="r")
        for name in ("current_force", "future_force")
    }
    train_abs_delta = np.abs(train_arrays["future_force"] - train_arrays["current_force"])
    dynamic_force_threshold = max(force_deadband, float(np.quantile(train_abs_delta, 0.75)))
    label_distributions = {}
    file_manifest = {}
    split_episode_sets = {}
    for split in ("train", "val", "test"):
        split_dir = args.cache_dir / split
        current_force = np.load(split_dir / "current_force.npy", mmap_mode="r")
        future_force = np.load(split_dir / "future_force.npy", mmap_mode="r")
        transition = np.load(split_dir / "contact_transition.npy", mmap_mode="r")
        dynamic = np.load(split_dir / "dynamic.npy", mmap_mode="r+")
        dynamic[:] = (transition != 0) & (transition != 2)
        dynamic[:] |= np.abs(future_force - current_force) > dynamic_force_threshold
        dynamic.flush()
        episode_ids = np.load(split_dir / "episode_id.npy", mmap_mode="r")
        split_episode_sets[split] = set(np.unique(episode_ids).tolist())
        trend = np.load(split_dir / "force_trend_class.npy", mmap_mode="r")
        finger = np.load(split_dir / "finger_change.npy", mmap_mode="r")
        label_distributions[split] = {
            "contact_transition": np.bincount(transition, minlength=4).tolist(),
            "force_trend": np.bincount(trend, minlength=3).tolist(),
            "per_finger_change": np.bincount(finger.reshape(-1), minlength=3).tolist(),
            "dynamic": int(np.asarray(dynamic).sum()),
            "dynamic_fraction": float(np.asarray(dynamic).mean()),
        }
        split_summaries[split]["dynamic_pairs"] = int(np.asarray(dynamic).sum())
        for path in sorted(split_dir.glob("*.npy")):
            array = np.load(path, mmap_mode="r")
            file_manifest[str(path.relative_to(args.cache_dir))] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "bytes": path.stat().st_size,
            }
    leakage = validate_episode_splits(
        {name: np.fromiter(values, dtype=np.int64) for name, values in split_episode_sets.items()}
    )
    if first_history is None:
        raise RuntimeError("no Teacher histories were generated")
    with torch.inference_mode():
        value = torch.from_numpy(first_history).to(device)
        deterministic_first = teacher.encode(value)
        deterministic_second = teacher.encode(value)
    deterministic = bool(torch.equal(deterministic_first, deterministic_second))
    teacher_digest_after = parameter_digest(teacher)
    teacher_unchanged = teacher_digest_before == teacher_digest_after
    if not deterministic or not teacher_unchanged:
        raise RuntimeError("frozen Teacher determinism/identity gate failed")
    manifest = {
        "schema": "tactile3d-unit.s2-transition-cache.v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_revision": s1_config["data"]["revision"],
        "public_config_sha256": sha256(args.config),
        "s1_config_sha256": sha256(args.s1_config),
        "s1_split_manifest_sha256": sha256(split_path),
        "s1_normalization_sha256": sha256(normalization_path),
        "teacher": {
            **teacher_identity,
            "checkpoint": str(args.teacher_checkpoint.resolve()),
            "parameter_sha256_before": teacher_digest_before,
            "parameter_sha256_after": teacher_digest_after,
            "parameters_changed": not teacher_unchanged,
            "eval_deterministic_exact": deterministic,
            "requires_grad": False,
        },
        "canonical_contract": TransitionPairContract(
            horizon_frames=CANONICAL_HORIZON
        ).to_dict(),
        "horizon_audit": horizon_summaries,
        "splits": split_summaries,
        "episode_leakage_counts": leakage,
        "thresholds": {
            "contact_public_sensor_units": contact_threshold,
            "force_trend_deadband_public_sensor_units": force_deadband,
            "dynamic_abs_force_delta_train_q75": dynamic_force_threshold,
            "fit_partition": "train",
        },
        "label_provenance": {
            "contact_transition": "DERIVED",
            "force_trend": "DERIVED",
            "per_finger_change": "DERIVED",
            "primitive": "ACTUAL METADATA",
            "object": "ACTUAL METADATA",
        },
        "label_distributions": label_distributions,
        "primitive_labels": primitive_labels,
        "object_labels": object_labels,
        "files": file_manifest,
        "build_seconds": time.monotonic() - start_time,
        "status": "PASS",
    }
    write_json(args.cache_dir / "manifest.json", manifest)
    write_json(args.output_dir / "transition_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "splits": split_summaries,
                "teacher_unchanged": teacher_unchanged,
                "deterministic": deterministic,
                "dynamic_force_threshold": dynamic_force_threshold,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
