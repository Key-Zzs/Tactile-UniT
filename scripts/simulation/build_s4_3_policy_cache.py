#!/usr/bin/env python3
"""Build and freeze the success-only S4.3 ACT train/dev feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_act import FrozenS42PolicyStack, PolicyNormalization  # noqa: E402
from gr00t.simulation.s4_3_training import (  # noqa: E402
    CACHE_ARRAYS,
    TASKS,
    atomic_json,
    cache_paths,
    canonical_sha256,
    create_array,
    read_json,
    sha256_file,
    valid_anchors,
)
from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb  # noqa: E402
from scripts.tactile_unit.build_c5_causal_visual_cache import (  # noqa: E402
    frozen_frame_features,
    verify_frozen_boundary_repeat,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_frozen_vision,
)

DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_3_policy_expert"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_3_restart"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
MEMBERSHIP = ROOT / ".local/artifacts/simulation/s4_3_pd/policy_expert_dataset_manifest.json"
WINDOWS = ROOT / ".local/artifacts/simulation/s4_3_pd/policy_window_counts.json"
VISION_IDENTITY = ROOT / ".local/artifacts/simulation/s4_2_formal/vision_identity.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=(
            Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-batch-size", type=int, default=64)
    parser.add_argument("--contact-batch-size", type=int, default=1024)
    parser.add_argument("--decode-workers", type=int, default=12)
    return parser.parse_args()


def episode_path(row: dict[str, Any]) -> Path:
    return DATASET_ROOT / row["attempt_relative_path"]


def load_episode(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = episode_path(row) / "steps.npz"
    if sha256_file(path) != row["steps_npz_sha256"]:
        raise RuntimeError(f"frozen episode checksum mismatch: {row['attempt_id']}")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def fit_normalization(rows: list[dict[str, Any]]) -> PolicyNormalization:
    proprio = []
    actions = []
    tactile = []
    for row in rows:
        values = load_episode(row)
        proprio.append(values["proprio"])
        actions.append(values["policy_action"])
        tactile.append(values["sim_tactile"])
    return PolicyNormalization.fit(
        np.concatenate(proprio), np.concatenate(actions), np.concatenate(tactile)
    )


def decode_current(item: tuple[Path, int]) -> np.ndarray:
    episode, t = item
    frame = cv2.imread(str(episode / "frames" / f"{t:06d}.jpg"), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"unreadable policy RGB: {episode.name}:{t}")
    return preprocess_trex_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def policy_pair_id(task: str, episode_id: str, t: int, split: str) -> bytes:
    value = f"{task}|{episode_id}|{t}|{t + 27}|POLICY_EXPERT_{split.upper()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest().encode("ascii")


def build_scope(
    task: str,
    split: str,
    rows: list[dict[str, Any]],
    vision: torch.nn.Module,
    stack: FrozenS42PolicyStack,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    membership_action_windows = sum(int(row["valid_bc_windows"]) for row in rows)
    expected = sum(len(valid_anchors(int(row["length"]))) for row in rows)
    paths = cache_paths(CACHE_ROOT, task, split)
    arrays = {
        name: create_array(paths[name], expected, tail, dtype)
        for name, (tail, dtype) in CACHE_ARRAYS.items()
    }
    vision_complete = create_array(
        paths["vision"].with_name("vision_complete.npy"), expected, (), np.bool_, fill=False
    )
    contact_complete = create_array(
        paths["contact_state"].with_name("contact_complete.npy"),
        expected,
        (),
        np.bool_,
        fill=False,
    )

    frame_items: list[tuple[Path, int]] = []
    cursor = 0
    episode_ids = []
    future_batches: list[np.ndarray] = []
    current_batch: list[np.ndarray] = []
    batch_rows: list[int] = []

    def flush_contact() -> None:
        nonlocal future_batches, current_batch, batch_rows
        if not batch_rows:
            return
        indices = np.asarray(batch_rows, dtype=np.int64)
        if not np.all(np.asarray(contact_complete[indices])):
            current = torch.from_numpy(np.stack(current_batch)).to(device)
            future = torch.from_numpy(np.stack(future_batches)).to(device)
            with torch.inference_mode():
                h = stack.encode_contact_state(current)
                target = stack.target_shared_contact(current, future)
            arrays["contact_state"][indices] = h.float().cpu().numpy()
            arrays["contact_target"][indices] = target.float().cpu().numpy()
            contact_complete[indices] = True
        future_batches, current_batch, batch_rows = [], [], []

    for episode_index, row in enumerate(rows):
        values = load_episode(row)
        length = len(values["control_step"])
        anchors = list(valid_anchors(length))
        if len(anchors) != int(row["valid_bc_windows"]) - 1:
            raise RuntimeError("full S4.2 transition window count mismatch")
        episode_ids.append(row["attempt_id"])
        directory = episode_path(row)
        for t in anchors:
            index = cursor
            arrays["proprio"][index] = values["proprio"][t]
            arrays["tactile_history"][index] = values["sim_tactile"][t - 25 : t + 1]
            arrays["action"][index] = values["policy_action"][t : t + 27]
            arrays["episode_index"][index] = episode_index
            arrays["t"][index] = t
            arrays["pair_id"][index] = policy_pair_id(task, row["attempt_id"], t, split)
            frame_items.append((directory, t))
            current_batch.append(values["sim_tactile"][t - 25 : t + 1])
            # Exact accepted S4.2 t -> t+27 timing convention.
            future_batches.append(values["sim_tactile"][t + 2 : t + 28])
            batch_rows.append(index)
            if len(batch_rows) == args.contact_batch_size:
                flush_contact()
            cursor += 1
    flush_contact()
    if cursor != expected or any(len(value) != expected for value in arrays.values()):
        raise RuntimeError("cache construction cardinality mismatch")

    incomplete = np.flatnonzero(~np.asarray(vision_complete))
    pool = ThreadPoolExecutor(max_workers=args.decode_workers)
    try:
        for start in range(0, len(incomplete), args.vision_batch_size):
            indices = incomplete[start : start + args.vision_batch_size]
            processed = np.stack(list(pool.map(decode_current, [frame_items[i] for i in indices])))
            if start == 0:
                verify_frozen_boundary_repeat(vision, processed[: min(4, len(processed))], device)
            features = frozen_frame_features(vision, processed, device)
            arrays["vision"][indices] = features
            vision_complete[indices] = True
            if start and start % (args.vision_batch_size * 100) == 0:
                print(
                    json.dumps({"task": task, "split": split, "vision_rows": int(start)}),
                    flush=True,
                )
    finally:
        pool.shutdown()

    for value in [*arrays.values(), vision_complete, contact_complete]:
        value.flush()
    if not np.asarray(vision_complete).all() or not np.asarray(contact_complete).all():
        raise RuntimeError("cache completion markers are incomplete")
    for name in (
        "vision",
        "proprio",
        "tactile_history",
        "contact_state",
        "action",
        "contact_target",
    ):
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise RuntimeError(f"non-finite cache array: {task}/{split}/{name}")
    if not np.array_equal(np.asarray(arrays["t"]), np.asarray([t for _, t in frame_items])):
        raise RuntimeError("cache timestep provenance mismatch")
    episode_payload = {"task": task, "split": split, "episode_ids": episode_ids}
    atomic_json(paths["vision"].with_name("episodes.json"), episode_payload)
    file_hashes = {name: sha256_file(path) for name, path in paths.items()}
    file_hashes["vision_complete"] = sha256_file(paths["vision"].with_name("vision_complete.npy"))
    file_hashes["contact_complete"] = sha256_file(paths["vision"].with_name("contact_complete.npy"))
    file_hashes["episodes"] = sha256_file(paths["vision"].with_name("episodes.json"))
    return {
        "rows": expected,
        "membership_action_only_windows": membership_action_windows,
        "excluded_for_full_t_plus_27_contact_target": membership_action_windows - expected,
        "episodes": len(rows),
        "pair_id_schema": "sha256(task,episode_id,t,t+27,POLICY_EXPERT_split)",
        "pair_id_first": arrays["pair_id"][0].decode("ascii"),
        "pair_id_last": arrays["pair_id"][-1].decode("ascii"),
        "shapes": {name: list(value.shape) for name, value in arrays.items()},
        "sha256": file_hashes,
        "future_vision_cached": False,
        "future_tactile_policy_input_cached": False,
        "status": "PASS",
    }


def main() -> None:
    args = parse_args()
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    device = torch.device(args.device)
    membership = read_json(MEMBERSHIP)
    windows = read_json(WINDOWS)
    if membership["status"] != "PASS" or windows["status"] != "PASS":
        raise RuntimeError("frozen policy membership is not ready")
    identity = read_json(VISION_IDENTITY)["checkpoint_file_sha256"]
    vision, vision_load = load_frozen_vision(
        args.unit_checkpoint,
        {"frozen_identity": {"original_unit_tokenizer_files_sha256": identity}},
        device,
    )
    if vision.training or vision_load["trainable_parameters"] != 0:
        raise RuntimeError("Original UniT Vision boundary is not frozen")
    stack = FrozenS42PolicyStack().to(device)
    if any(parameter.requires_grad for parameter in stack.parameters()):
        raise RuntimeError("S4.2 stack is not frozen")

    by_task_split = {
        task: {
            "train": [row for row in membership["successful_train"] if row["task"] == task],
            "dev": [row for row in membership["successful_dev"] if row["task"] == task],
        }
        for task in TASKS
    }
    manifest: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-policy-training-cache.v1",
        "stage": "R8",
        "membership_manifest_sha256": sha256_file(MEMBERSHIP),
        "policy_window_counts_sha256": sha256_file(WINDOWS),
        "vision_identity_sha256": sha256_file(VISION_IDENTITY),
        "vision_current_frame_only": True,
        "contact_target_timing": "t -> t+27; future tactile indices t+2:t+27 inclusive",
        "normalization_fit": "all timesteps of success-only POLICY_EXPERT_TRAIN episodes per task",
        "tasks": {},
        "immutable": True,
        "status": "BUILDING",
    }
    for task in TASKS:
        normalization = fit_normalization(by_task_split[task]["train"])
        normalization_payload = {
            "schema": "tactile3d-unit.s4-3-policy-normalization.v1",
            "task": task,
            "fit_split": "POLICY_EXPERT_TRAIN",
            "dev_statistics_used": False,
            "statistics": normalization.to_json(),
        }
        normalization_path = CACHE_ROOT / task / "normalization.json"
        atomic_json(normalization_path, normalization_payload)
        manifest["tasks"][task] = {
            "normalization_sha256": sha256_file(normalization_path),
        }
        for split in ("train", "dev"):
            print(json.dumps({"task": task, "split": split, "status": "START"}), flush=True)
            manifest["tasks"][task][split] = build_scope(
                task, split, by_task_split[task][split], vision, stack, device, args
            )
            atomic_json(CACHE_ROOT / "manifest.building.json", manifest)
    manifest["cache_content_sha256"] = canonical_sha256(manifest["tasks"])
    manifest["status"] = "PASS"
    atomic_json(CACHE_ROOT / "manifest.json", manifest)
    audit = {**manifest, "cache_root": ".local/cache/simulation/s4_3_restart"}
    atomic_json(ARTIFACT_ROOT / "policy_training_cache.json", audit)
    print(
        json.dumps(
            {
                "rows": sum(
                    s[x]["rows"] for s in manifest["tasks"].values() for x in ("train", "dev")
                ),
                "status": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
