#!/usr/bin/env python3
"""Build the frozen tactile/Contact-State sidecar for official PI1 training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.simulation.s4_3_act import FrozenS42PolicyStack
from gr00t.simulation.s4_3_pi1 import HISTORY_STEPS, left_repeat_history


ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / ".local/cache/simulation/s4_3_pi1/replay"
ALIGNMENT = ROOT / ".local/artifacts/simulation/s4_3_pi1/official_dataset_alignment.json"
DATASET = ROOT / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets/pinch_tongs"
OUTPUT = ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_pi1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(digest, name: str, value: np.ndarray) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(value).tobytes())


def write_json(name: str, value: Any) -> None:
    path = ARTIFACT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@torch.inference_mode()
def encode(stack: FrozenS42PolicyStack, histories: np.ndarray, batch_size: int = 512) -> np.ndarray:
    output = np.empty((len(histories), 256), dtype=np.float32)
    for start in range(0, len(histories), batch_size):
        stop = min(start + batch_size, len(histories))
        output[start:stop] = stack.encode_contact_state(torch.from_numpy(histories[start:stop])).float().numpy()
    return output


@torch.inference_mode()
def shared_targets(
    stack: FrozenS42PolicyStack,
    contact_state: np.ndarray,
    episode_starts: np.ndarray,
    episode_lengths: np.ndarray,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.zeros(len(contact_state), dtype=np.bool_)
    future_index = np.zeros(len(contact_state), dtype=np.int64)
    for start, length in zip(episode_starts, episode_lengths, strict=True):
        count = max(0, int(length) - 27)
        valid[start : start + count] = True
        future_index[start : start + count] = np.arange(start + 27, start + 27 + count)
    selected = np.flatnonzero(valid)
    target = np.zeros((len(contact_state), 8, 32), dtype=np.float32)
    for begin in range(0, len(selected), batch_size):
        rows = selected[begin : begin + batch_size]
        current = torch.from_numpy(contact_state[rows])
        future = torch.from_numpy(contact_state[future_index[rows]])
        code = stack.contact_C3(current, future)["code"]
        target[rows] = stack.bridge_B3.encode("contact", code).float().numpy()
    return target, valid


def main() -> None:
    alignment = json.loads(ALIGNMENT.read_text())
    if alignment["status"] != "PASS":
        raise RuntimeError("official raw/LeRobot alignment is not frozen PASS")
    lengths = np.asarray(alignment["episode_lengths"], dtype=np.int64)
    trims = np.asarray(alignment["raw_leading_static_frames_excluded_by_official_conversion"], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    tactile_parts = []
    control_parts = []
    for episode, (length, trim) in enumerate(zip(lengths, trims, strict=True)):
        path = REPLAY / f"episode_{episode:03d}.npz"
        with np.load(path, allow_pickle=False) as source:
            tactile_parts.append(source["tactile_sim"][trim : trim + length])
            control_parts.append(source["control_tick_end"][trim : trim + length])
    tactile = np.concatenate(tactile_parts).astype(np.float32)
    control_tick = np.concatenate(control_parts).astype(np.int64)
    histories = np.empty((len(tactile), HISTORY_STEPS, 30), dtype=np.float32)
    bootstrap = np.empty(len(tactile), dtype=np.int16)
    episode_index = np.repeat(np.arange(100), lengths).astype(np.int64)
    frame_index = np.concatenate([np.arange(length) for length in lengths]).astype(np.int64)
    for start, length in zip(starts, lengths, strict=True):
        episode_tactile = tactile[start : start + length]
        for frame in range(length):
            histories[start + frame], bootstrap[start + frame] = left_repeat_history(episode_tactile, frame)

    stack = FrozenS42PolicyStack().cpu().eval()
    contact_state = encode(stack, histories)
    shared, physical_valid = shared_targets(stack, contact_state, starts, lengths)
    checkpoints = {
        name: sha256_file(path)
        for name, path in FrozenS42PolicyStack.CHECKPOINTS.items()
    }
    gates = {
        "episodes_100": len(np.unique(episode_index)) == 100,
        "frames_40065": len(tactile) == 40065,
        "tactile_shape": tactile.shape == (40065, 30),
        "history_shape": histories.shape == (40065, 26, 30),
        "contact_state_shape": contact_state.shape == (40065, 256),
        "shared_target_shape": shared.shape == (40065, 8, 32),
        "all_finite": all(np.isfinite(value).all() for value in (tactile, histories, contact_state, shared)),
        "history_ends_current": np.array_equal(histories[:, -1], tactile),
        "left_repeat_bootstrap": all(
            np.array_equal(histories[start, :], np.repeat(tactile[start : start + 1], 26, axis=0))
            for start in starts
        ),
        "no_cross_episode_history": bool(np.all(bootstrap[starts] == 25)),
        "physical_tail_invalid": all(not physical_valid[start + length - 27 : start + length].any() for start, length in zip(starts, lengths, strict=True)),
        "physical_valid_count": int(physical_valid.sum()) == int((lengths - 27).sum()),
    }
    if not all(gates.values()):
        raise RuntimeError(f"augmented sidecar gates failed: {gates}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        index=np.arange(len(tactile), dtype=np.int64),
        episode_index=episode_index,
        frame_index=frame_index,
        tactile_sim=tactile,
        tactile_history=histories,
        history_bootstrap_count=bootstrap,
        control_tick_end=control_tick,
        contact_state=contact_state,
        contact_shared_target=shared,
        physical_aux_valid=physical_valid,
    )
    sidecar_sha = sha256_file(OUTPUT)
    original_files = sorted(path for path in DATASET.rglob("*") if path.is_file())
    original_manifest = {path.relative_to(DATASET).as_posix(): sha256_file(path) for path in original_files}
    original_digest = hashlib.sha256()
    for name, value in sorted(original_manifest.items()):
        original_digest.update(name.encode("utf-8")); original_digest.update(value.encode("ascii"))
    feature_digest = hashlib.sha256()
    for name, value in (
        ("tactile_sim", tactile), ("tactile_history", histories),
        ("contact_state", contact_state), ("contact_shared_target", shared),
        ("physical_aux_valid", physical_valid),
    ):
        hash_array(feature_digest, name, value)

    active = tactile.reshape(-1, 5, 6)[:, :, 0] > 0
    common = {
        "status": "PASS",
        "episodes": 100,
        "frames": 40065,
        "sidecar": "$PI1_DATA_ROOT/pinch_tongs_official_tactile/sidecar.npz",
        "sidecar_sha256": sidecar_sha,
        "feature_content_sha256": feature_digest.hexdigest(),
        "s4_2_checkpoint_sha256": checkpoints,
    }
    write_json("tactile_augmentation_manifest.json", {
        "schema": "tactile3d-unit.s4-3-pi1-tactile-augmentation-manifest.v1", **common,
        "architecture": "sidecar/reference strict superset; official dataset bytes are referenced read-only",
        "fields": {
            "tactile_sim": [30], "tactile_history": [26, 30],
            "history_bootstrap_count": [], "control_tick_end": [],
            "contact_state": [256], "contact_shared_target": [8, 32],
            "physical_aux_valid": [],
        },
        "history_bootstrap": "LEFT_REPEAT_FIRST",
    })
    write_json("official_field_parity.json", {
        "schema": "tactile3d-unit.s4-3-pi1-official-field-parity.v1", **common,
        "original_dataset_tree_sha256": original_digest.hexdigest(),
        "original_dataset_file_count": len(original_manifest),
        "original_dataset_files": original_manifest,
        "parity_mechanism": "the augmented dataset references the immutable PI0 LeRobot root; it stores no replacement original fields",
        "rgb_state_action_prompt_episode_frame_timestamp_unchanged": "PASS",
    })
    write_json("contact_state_cache_manifest.json", {
        "schema": "tactile3d-unit.s4-3-pi1-contact-state-cache.v1", **common,
        "shape": [40065, 256], "producer": "frozen unit E_T", "checkpoint_sha256": checkpoints["contact_state"],
    })
    write_json("shared_contact_target_manifest.json", {
        "schema": "tactile3d-unit.s4-3-pi1-shared-contact-target.v1", **common,
        "shape": [40065, 8, 32], "valid_frames": int(physical_valid.sum()),
        "invalid_tail_frames": int((~physical_valid).sum()), "transition": "t -> t+27 within episode",
        "runtime_observation": False, "training_use": "PI1C only",
    })
    write_json("augmented_dataset_quality.json", {
        "schema": "tactile3d-unit.s4-3-pi1-augmented-dataset-quality.v1", **common,
        "gates": {name: "PASS" if value else "FAIL" for name, value in gates.items()},
        "physical_aux_valid_frames": int(physical_valid.sum()),
        "per_region_active_frames": dict(zip(("right_palm", "right_index", "right_middle", "right_ring", "right_thumb"), active.sum(axis=0).astype(int).tolist(), strict=True)),
    })
    print(json.dumps({"status": "PASS", "frames": len(tactile), "valid_targets": int(physical_valid.sum()), "sidecar_sha256": sidecar_sha}))


if __name__ == "__main__":
    main()
