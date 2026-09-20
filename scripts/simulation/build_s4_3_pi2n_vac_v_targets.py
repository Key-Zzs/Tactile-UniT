#!/usr/bin/env python3
"""Build the frozen Vision-side VAC target cache at exact control-row offset +27."""

from __future__ import annotations

import argparse
from collections import deque
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_formal import FormalVACBridge  # noqa: E402
from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb  # noqa: E402
from scripts.simulation.build_s4_3_pi2u_bva_targets import (  # noqa: E402
    DATASET_ROOT,
    UNIT_ROOT,
    episode_frames,
    load_front_pointers,
    sha256_file,
)
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision  # noqa: E402


PROTOCOL = ROOT / "configs/simulation/s4_3_pi2n_vac_v_target_protocol.json"
BRIDGE_PATH = ROOT / ".local/experiments/simulation/s4_2_formal/bridge/selected.pt"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_3_pi2n"
TARGET_OFFSET_ROWS = 27
CONTROL_DT_SECONDS = 0.02
PHYSICAL_HORIZON_SECONDS = TARGET_OFFSET_ROWS * CONTROL_DT_SECONDS


def run_root() -> Path:
    configured = os.environ.get("PI2N_RUN_ROOT")
    path = Path(configured) if configured else ROOT / ".local/experiments/simulation/s4_3_pi2n"
    resolved = path.resolve()
    expected = Path("/mnt/ugreen_nas/storage/UniT_storage/experiments/simulation/s4_3_pi2n")
    if resolved != expected:
        raise RuntimeError(f"PI2N run root does not resolve to the audited NAS path: {resolved}")
    return resolved


def output_path() -> Path:
    return run_root() / "caches/pinch_tongs_vac_v_t27/sidecar.npz"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def expected_valid_mask(lengths: list[int] | np.ndarray) -> np.ndarray:
    parts = []
    for length in map(int, lengths):
        if length <= TARGET_OFFSET_ROWS:
            raise ValueError(f"episode length {length} cannot support +{TARGET_OFFSET_ROWS}")
        value = np.zeros(length, dtype=np.bool_)
        value[: length - TARGET_OFFSET_ROWS] = True
        parts.append(value)
    return np.concatenate(parts)


def validate_protocol(protocol: dict[str, Any]) -> None:
    target = protocol["target"]
    dataset = protocol["policy_dataset"]
    teacher = protocol["teacher"]
    if protocol["status"] != "FROZEN_BEFORE_TARGET_BUILD":
        raise RuntimeError("VAC-V target protocol is not frozen")
    if target["future_row_offset"] != TARGET_OFFSET_ROWS:
        raise RuntimeError("VAC-V target row offset changed")
    if target["physical_horizon_seconds"] != PHYSICAL_HORIZON_SECONDS:
        raise RuntimeError("VAC-V physical horizon changed")
    if target["future_selection"] != "same episode exact row i+27; no interpolation":
        raise RuntimeError("VAC-V future selection changed")
    if dataset["expected_valid_rows"] != 37365 or dataset["expected_invalid_rows"] != 2700:
        raise RuntimeError("VAC-V target row accounting changed")
    if sha256_file(BRIDGE_PATH) != teacher["bridge_checkpoint_sha256"]:
        raise RuntimeError("frozen B3 VAC bridge hash mismatch")
    for name, expected in teacher["official_unit_files_sha256"].items():
        if sha256_file(UNIT_ROOT / "tokenizer" / name) != expected:
            raise RuntimeError(f"official UniT tokenizer hash mismatch: {name}")
    if sha256_file(DATASET_ROOT / "meta/info.json") != dataset["info_sha256"]:
        raise RuntimeError("policy dataset info hash mismatch")
    episode_tables = sorted((DATASET_ROOT / "meta/episodes").rglob("*.parquet"))
    if len(episode_tables) != 1 or sha256_file(episode_tables[0]) != dataset["episode_table_sha256"]:
        raise RuntimeError("policy dataset episode table hash mismatch")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = output_path()
    manifest_path = ARTIFACTS / "vac_v_target_manifest.json"
    if output.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite frozen VAC-V target output: {output}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("set CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("VAC-V target construction requires exactly one explicitly visible GPU")

    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    pointers = load_front_pointers()
    lengths = [pointer.length for pointer in pointers]
    total = sum(lengths)
    valid_expected = expected_valid_mask(lengths)
    if len(pointers) != 100 or total != 40065:
        raise RuntimeError("policy episode/row identity mismatch")
    if int(valid_expected.sum()) != 37365 or int((~valid_expected).sum()) != 2700:
        raise RuntimeError("expected +27 validity accounting mismatch")

    device = torch.device("cuda:0")
    spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": protocol["teacher"]["official_unit_files_sha256"]}}
    vision, loading = load_frozen_vision(UNIT_ROOT, spec, device)
    checkpoint = torch.load(BRIDGE_PATH, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("candidate") != protocol["teacher"]["bridge_checkpoint_candidate"]
        or checkpoint.get("adapter") != protocol["teacher"]["bridge_adapter"]
    ):
        raise RuntimeError("selected bridge checkpoint metadata differs from frozen B3 identity")
    bridge = FormalVACBridge("slot").eval().requires_grad_(False).to(device)
    bridge.load_state_dict(checkpoint["state_dict"], strict=True)

    target = np.zeros((total, 8, 32), dtype=np.float32)
    valid = np.zeros(total, dtype=np.bool_)
    indices = np.arange(total, dtype=np.int64)
    pending_current: list[np.ndarray] = []
    pending_future: list[np.ndarray] = []
    pending_index: list[int] = []
    maximum_decode_timestamp_error = 0.0
    started = time.monotonic()

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
            raise RuntimeError("non-finite or malformed VAC-V target batch")
        target[np.asarray(pending_index)] = values
        valid[np.asarray(pending_index)] = True
        pending_current.clear()
        pending_future.clear()
        pending_index.clear()

    for episode_number, pointer in enumerate(pointers):
        window: deque[np.ndarray] = deque(maxlen=TARGET_OFFSET_ROWS + 1)
        path = DATASET_ROOT / pointer.relative_path
        count = 0
        for frame_index, rgb, timestamp in episode_frames(path, pointer):
            count += 1
            expected_video_timestamp = pointer.from_timestamp + frame_index / 30.0
            maximum_decode_timestamp_error = max(maximum_decode_timestamp_error, abs(timestamp - expected_video_timestamp))
            window.append(preprocess_trex_rgb(rgb))
            if frame_index >= TARGET_OFFSET_ROWS:
                anchor = pointer.dataset_from_index + frame_index - TARGET_OFFSET_ROWS
                pending_current.append(window[0])
                pending_future.append(window[-1])
                pending_index.append(anchor)
                if len(pending_index) >= args.batch_size:
                    flush()
        if count != pointer.length:
            raise RuntimeError(f"episode {pointer.episode_id} frame count mismatch")
        flush()
        elapsed = time.monotonic() - started
        done = episode_number + 1
        print(json.dumps({
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "episode": done,
            "episodes": len(pointers),
            "targets": int(valid.sum()),
            "elapsed_seconds": elapsed,
            "eta_seconds": elapsed / done * (len(pointers) - done),
        }, sort_keys=True), flush=True)

    if not np.array_equal(valid, valid_expected):
        raise RuntimeError("generated VAC-V validity differs from exact episode-local +27 mask")
    if np.any(target[~valid] != 0) or not np.isfinite(target).all():
        raise RuntimeError("VAC-V target finiteness/invalid-placeholder contract failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, index=indices, vac_vision_target=target, vac_aux_valid=valid)
    temporary.replace(output)

    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2n-vac-v-target-manifest.v1",
        "status": "BUILT_PENDING_PARITY_AUDIT",
        "model_id": "B_VAC_V",
        "sidecar": "$PI2N_RUN_ROOT/caches/pinch_tongs_vac_v_t27/sidecar.npz",
        "sidecar_realpath": str(output),
        "sidecar_sha256": sha256_file(output),
        "fields": ["index", "vac_vision_target", "vac_aux_valid"],
        "rows": total,
        "valid_rows": int(valid.sum()),
        "invalid_tail_rows": int((~valid).sum()),
        "episodes": len(pointers),
        "target_shape": [8, 32],
        "source_future_offset_rows": TARGET_OFFSET_ROWS,
        "source_control_dt_seconds": CONTROL_DT_SECONDS,
        "physical_horizon_seconds": PHYSICAL_HORIZON_SECONDS,
        "source_frame_selection": "same episode exact row i+27; no interpolation",
        "maximum_absolute_decode_timestamp_error_seconds": maximum_decode_timestamp_error,
        "official_unit_loading": loading,
        "official_unit_checkpoint_revision": protocol["teacher"]["official_unit_checkpoint_revision"],
        "vac_bridge_sha256": sha256_file(BRIDGE_PATH),
        "vac_bridge_role": protocol["teacher"]["bridge_role"],
        "vac_bridge_checkpoint_candidate": checkpoint["candidate"],
        "vac_bridge_adapter": checkpoint["adapter"],
        "target_inputs": ["observation.images.front[current_row]", "observation.images.front[future_row_i_plus_27]"],
        "target_projector": "FormalVACBridge.encode(vision)",
        "contact_tactile_fields_read": [],
        "clean_VA_projector_used": False,
        "historical_artifacts_modified": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "sidecar_sha256": manifest["sidecar_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
