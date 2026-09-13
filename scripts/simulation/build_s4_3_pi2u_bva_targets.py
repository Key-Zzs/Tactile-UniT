#!/usr/bin/env python3
"""Build the contact-free canonical 0.54 s future-Vision target sidecar for BVA.

The simulator representation clock is 50 Hz, while the official policy dataset
is natively sampled at 30 Hz.  Canonical step ``t+27`` is therefore represented
by the nearest native observation, dataset frame ``i+16`` (0.533333... s).  The
6.667 ms discrepancy is the frozen S4.1 sampling-rounding error.  Keeping a
native frame avoids inventing an interpolated RGB observation.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator

import av
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (
    ROOT
    / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/"
    "dexjoco_lerobot_datasets/pinch_tongs"
)
UNIT_ROOT = ROOT / ".local/external/s4_3_pi2u/unit_fulldata/VLA-UniT-3B-fulldata"
BRIDGE_PATH = ROOT / ".local/experiments/simulation/s4_3_pi2u/va_bridge/frozen.pt"
PROTOCOL = ROOT / "configs/simulation/s4_3_pi2u_bva_protocol.json"
OUTPUT = ROOT / ".local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2u"
CANONICAL_CONTROL_DT_SECONDS = 0.02
CANONICAL_HORIZON_STEPS = 27
CANONICAL_HORIZON_SECONDS = CANONICAL_HORIZON_STEPS * CANONICAL_CONTROL_DT_SECONDS
SOURCE_FPS = 30.0
SOURCE_HORIZON_FRAMES = round(CANONICAL_HORIZON_SECONDS * SOURCE_FPS)
SOURCE_HORIZON_SECONDS = SOURCE_HORIZON_FRAMES / SOURCE_FPS
SOURCE_TIMING_ERROR_SECONDS = abs(CANONICAL_HORIZON_SECONDS - SOURCE_HORIZON_SECONDS)


@dataclass(frozen=True)
class EpisodePointer:
    episode_id: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    from_timestamp: float
    to_timestamp: float
    relative_path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def contact_leakage_payload() -> dict[str, Any]:
    return {
        "schema": "tactile3d-unit.s4-3-pi2u-contact-leakage-audit.v1",
        "status": "PASS",
        "sidecar_fields_exact": sorted(["index", "va_shared_target", "va_aux_valid"]),
        "contact_tactile_force_touch_fields": [],
        "target_lineage": [
            "observation.images.front[t]",
            "observation.images.front[nearest native frame to canonical t+27]",
            "30 Hz dataset frame i+16 (0.533333 s; 0.006667 s from canonical 0.54 s)",
            "frozen official UniT Vision transition encoder",
            "frozen VA-only bridge vision projector",
            "u_v[8,32]",
        ],
        "runtime_inputs_added_to_policy": [],
        "invalid_tail_placeholder_depends_on_contact": False,
        "outcome_or_rollout_data_used": False,
    }


def load_front_pointers() -> list[EpisodePointer]:
    """Read the generic LeRobot v3 episode table without task-specific columns."""

    import pyarrow.parquet as pq

    info = json.loads((DATASET_ROOT / "meta/info.json").read_text())
    prefix = "videos/observation.images.front"
    columns = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    pointers = []
    for path in sorted((DATASET_ROOT / "meta/episodes").rglob("*.parquet")):
        values = pq.read_table(path, columns=columns).to_pydict()
        for row in range(len(values["episode_index"])):
            chunk = int(values[f"{prefix}/chunk_index"][row])
            file_index = int(values[f"{prefix}/file_index"][row])
            pointers.append(
                EpisodePointer(
                    episode_id=int(values["episode_index"][row]),
                    length=int(values["length"][row]),
                    dataset_from_index=int(values["dataset_from_index"][row]),
                    dataset_to_index=int(values["dataset_to_index"][row]),
                    from_timestamp=float(values[f"{prefix}/from_timestamp"][row]),
                    to_timestamp=float(values[f"{prefix}/to_timestamp"][row]),
                    relative_path=info["video_path"].format(
                        video_key="observation.images.front",
                        chunk_index=chunk,
                        file_index=file_index,
                    ),
                )
            )
    pointers.sort(key=lambda item: item.episode_id)
    if [item.episode_id for item in pointers] != list(range(int(info["total_episodes"]))):
        raise RuntimeError("episode IDs are not unique and contiguous")
    if any(item.dataset_to_index - item.dataset_from_index != item.length for item in pointers):
        raise RuntimeError("episode dataset interval mismatch")
    return pointers


def episode_frames(path: Path, pointer) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield one nearest decoded RGB frame for each episode-relative index."""

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if pointer.from_timestamp > 0 and stream.time_base is not None:
            container.seek(
                int(max(0.0, pointer.from_timestamp - 2.0 / SOURCE_FPS) / float(stream.time_base)),
                stream=stream,
                any_frame=False,
                backward=True,
            )
        selected: dict[int, tuple[float, Any, float]] = {}
        final_time = pointer.from_timestamp + (pointer.length - 1) / SOURCE_FPS
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < pointer.from_timestamp - 0.51 / SOURCE_FPS:
                continue
            if timestamp > final_time + 0.51 / SOURCE_FPS:
                break
            index = int(round((timestamp - pointer.from_timestamp) * SOURCE_FPS))
            if not 0 <= index < pointer.length:
                continue
            distance = abs(timestamp - (pointer.from_timestamp + index / SOURCE_FPS))
            previous = selected.get(index)
            if previous is None or distance < previous[0]:
                selected[index] = (distance, frame, timestamp)
        if sorted(selected) != list(range(pointer.length)):
            missing = sorted(set(range(pointer.length)) - set(selected))
            raise RuntimeError(f"episode {pointer.episode_id} decoded frame gaps: {missing[:8]}")
        for index in range(pointer.length):
            _, frame, timestamp = selected[index]
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (640, 640, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"unexpected RGB frame contract: {rgb.shape} {rgb.dtype}")
            yield index, rgb, timestamp


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if OUTPUT.exists():
        manifest_path = ARTIFACTS / "bva_target_manifest.json"
        leakage_path = ARTIFACTS / "contact_leakage_audit.json"
        with np.load(OUTPUT, allow_pickle=False) as source:
            exact_fields = set(source.files) == {"index", "va_shared_target", "va_aux_valid"}
        if manifest_path.is_file() and exact_fields and not leakage_path.exists():
            atomic_json(leakage_path, contact_leakage_payload())
            print(json.dumps({"status": "RECOVERED_POST_BUILD_AUDIT", "sidecar_sha256": sha256_file(OUTPUT)}, sort_keys=True))
            return
        raise SystemExit(f"refusing to overwrite frozen BVA sidecar: {OUTPUT}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("BVA target construction requires exactly one explicitly visible GPU")
    protocol = json.loads(PROTOCOL.read_text())
    temporal = protocol["auxiliary_target"]["temporal_alignment"]
    expected_temporal = {
        "canonical_control_dt_seconds": CANONICAL_CONTROL_DT_SECONDS,
        "canonical_offset_steps": CANONICAL_HORIZON_STEPS,
        "canonical_horizon_seconds": CANONICAL_HORIZON_SECONDS,
        "source_dataset_fps": SOURCE_FPS,
        "source_offset_frames": SOURCE_HORIZON_FRAMES,
        "source_horizon_seconds": SOURCE_HORIZON_SECONDS,
        "absolute_timing_error_seconds": SOURCE_TIMING_ERROR_SECONDS,
        "selection_rule": "nearest native source frame; no RGB interpolation",
    }
    if temporal != expected_temporal:
        raise RuntimeError("BVA temporal-alignment protocol mismatch")
    for name, expected in protocol["official_unit"]["files_sha256"].items():
        actual = sha256_file(UNIT_ROOT / "tokenizer" / name)
        if actual != expected:
            raise RuntimeError(f"official UniT tokenizer hash mismatch: {name}")
    if sha256_file(BRIDGE_PATH) != protocol["va_bridge"]["checkpoint_sha256"]:
        raise RuntimeError("VA bridge checkpoint hash mismatch")

    sys.path.insert(0, str(ROOT))
    from gr00t.simulation.s4_3_pi2u_va import VAOnlyBridge
    from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb
    from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision

    device = torch.device("cuda:0")
    spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": protocol["official_unit"]["files_sha256"]}}
    vision, loading = load_frozen_vision(UNIT_ROOT, spec, device)
    checkpoint = torch.load(BRIDGE_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("modalities") != ["vision", "action"]:
        raise RuntimeError("VA bridge modality contract mismatch")
    bridge = VAOnlyBridge().eval().requires_grad_(False).to(device)
    bridge.load_state_dict(checkpoint["state_dict"], strict=True)

    pointers = load_front_pointers()
    total = sum(pointer.length for pointer in pointers)
    target = np.zeros((total, 8, 32), dtype=np.float32)
    valid = np.zeros(total, dtype=np.bool_)
    indices = np.arange(total, dtype=np.int64)
    pending_current: list[np.ndarray] = []
    pending_future: list[np.ndarray] = []
    pending_index: list[int] = []
    maximum_timestamp_error = 0.0

    def flush() -> None:
        if not pending_index:
            return
        obs = torch.from_numpy(np.stack(pending_current))[:, None].to(device, dtype=vision.dtype)
        goal = torch.from_numpy(np.stack(pending_future))[:, None].to(device, dtype=vision.dtype)
        transition, _, _ = vision.vision_branch(obs, goal, batch_size=len(pending_index))
        native = vision.vq_down_resampler(transition)
        shared = bridge.encode("vision", native.float())
        values = shared.float().cpu().numpy()
        if values.shape != (len(pending_index), 8, 32) or not np.isfinite(values).all():
            raise RuntimeError("non-finite or malformed BVA target batch")
        target[np.asarray(pending_index)] = values
        valid[np.asarray(pending_index)] = True
        pending_current.clear()
        pending_future.clear()
        pending_index.clear()

    for episode_number, pointer in enumerate(pointers):
        window: deque[np.ndarray] = deque(maxlen=SOURCE_HORIZON_FRAMES + 1)
        path = DATASET_ROOT / pointer.relative_path
        count = 0
        for frame_index, rgb, timestamp in episode_frames(path, pointer):
            count += 1
            expected_time = pointer.from_timestamp + frame_index / SOURCE_FPS
            maximum_timestamp_error = max(maximum_timestamp_error, abs(timestamp - expected_time))
            window.append(preprocess_trex_rgb(rgb))
            if frame_index >= SOURCE_HORIZON_FRAMES:
                anchor = pointer.dataset_from_index + frame_index - SOURCE_HORIZON_FRAMES
                pending_current.append(window[0])
                pending_future.append(window[-1])
                pending_index.append(anchor)
                if len(pending_index) >= args.batch_size:
                    flush()
        if count != pointer.length:
            raise RuntimeError(f"episode {pointer.episode_id} frame count mismatch")
        flush()
        print(f"episode {episode_number + 1:03d}/{len(pointers):03d} targets={int(valid.sum())}", flush=True)

    expected_valid = sum(pointer.length - SOURCE_HORIZON_FRAMES for pointer in pointers)
    expected_invalid = SOURCE_HORIZON_FRAMES * len(pointers)
    if int(valid.sum()) != expected_valid or int((~valid).sum()) != expected_invalid:
        raise RuntimeError("BVA validity accounting mismatch")
    if np.any(target[~valid] != 0) or not np.isfinite(target).all():
        raise RuntimeError("invalid BVA rows must be finite all-zero placeholders")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, index=indices, va_shared_target=target, va_aux_valid=valid)
    temporary.replace(OUTPUT)

    source_files = [DATASET_ROOT / "meta/info.json", *sorted((DATASET_ROOT / "meta/episodes").rglob("*.parquet"))]
    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2u-bva-target-manifest.v1",
        "status": "PASS",
        "sidecar": "$REPO_ROOT/.local/datasets/simulation/s4_3_pi2u/pinch_tongs_va/sidecar.npz",
        "sidecar_sha256": sha256_file(OUTPUT),
        "fields": ["index", "va_shared_target", "va_aux_valid"],
        "rows": total,
        "valid_rows": int(valid.sum()),
        "invalid_tail_rows": int((~valid).sum()),
        "episodes": len(pointers),
        "target_shape": [8, 32],
        "canonical_future_offset_steps": CANONICAL_HORIZON_STEPS,
        "canonical_future_offset_seconds": CANONICAL_HORIZON_SECONDS,
        "source_future_offset_frames": SOURCE_HORIZON_FRAMES,
        "source_future_offset_seconds": SOURCE_HORIZON_SECONDS,
        "absolute_timing_error_seconds": SOURCE_TIMING_ERROR_SECONDS,
        "source_frame_selection": "nearest native source frame; no RGB interpolation",
        "camera": "observation.images.front",
        "maximum_absolute_decode_timestamp_error_seconds": maximum_timestamp_error,
        "dataset_contract_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_files},
        "official_unit_loading": loading,
        "official_unit_checkpoint_revision": protocol["official_unit"]["checkpoint_revision"],
        "va_bridge_sha256": sha256_file(BRIDGE_PATH),
        "contact_fields_present": [],
    }
    atomic_json(ARTIFACTS / "bva_target_manifest.json", manifest)
    atomic_json(ARTIFACTS / "contact_leakage_audit.json", contact_leakage_payload())
    print(json.dumps({"status": "PASS", "rows": total, "valid": int(valid.sum()), "sha256": manifest["sidecar_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
