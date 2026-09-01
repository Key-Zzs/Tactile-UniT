#!/usr/bin/env python3
"""Build exact DS-TRAIN/DEV Vision, Action, and Contact pilot pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.paired_contract import preprocess_trex_rgb, sha256_file  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_frozen_vision,
)


PAIR_PATH = ROOT / ".local/cache/simulation/s4_2/pairs/train.npz"
CONTACT_PATH = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics/train.npz"
CONTACT_CODE_PATH = ROOT / ".local/cache/simulation/s4_2ds/contact_candidates.npz"
ACTION_PATH = ROOT / ".local/cache/simulation/s4_2ds/action_pilot.npz"
CONFIG_PATH = ROOT / "configs/simulation/s4_2ds_representation_selection.json"
PROTOCOL_PATH = ROOT / ".local/artifacts/simulation/s4_2ds/protocol_freeze.json"
ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_2ds"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2ds"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_frame(path_value: str) -> np.ndarray:
    path = Path(path_value)
    with Image.open(path) as image:
        return preprocess_trex_rgb(np.asarray(image.convert("RGB"), dtype=np.uint8))


@torch.inference_mode()
def encode_batch(
    vision: torch.nn.Module,
    current_paths: np.ndarray,
    future_paths: np.ndarray,
    executor: ThreadPoolExecutor,
) -> np.ndarray:
    current = list(executor.map(read_frame, current_paths.tolist()))
    future = list(executor.map(read_frame, future_paths.tolist()))
    obs = torch.from_numpy(np.stack(current))[:, None].to(vision.device, dtype=vision.dtype)
    goal = torch.from_numpy(np.stack(future))[:, None].to(vision.device, dtype=vision.dtype)
    value, _, _ = vision.vision_branch(obs, goal, batch_size=len(current))
    return vision.vq_down_resampler(value).float().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unit-checkpoint",
        type=Path,
        default=Path(os.environ["UNIT_FULLDATA_CKPT"])
        if os.environ.get("UNIT_FULLDATA_CKPT")
        else None,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("S4.2-DS protocol changed after freeze")
    if json.loads(ARTIFACT_ROOT.joinpath("action_pilot.json").read_text())["overall"] != "PASS":
        raise RuntimeError("Action pilot did not pass")
    pairs = load_npz(PAIR_PATH)
    contact = load_npz(CONTACT_PATH)
    contact_codes = load_npz(CONTACT_CODE_PATH)
    action = load_npz(ACTION_PATH)
    for source_name, source in (
        ("Contact-State cache", contact),
        ("Contact candidates", contact_codes),
        ("Action pilot", action),
    ):
        if not np.array_equal(pairs["pair_id"], source["pair_id"]):
            raise RuntimeError(f"pair identity mismatch: {source_name}")
    expected = config["vision_pilot"]["checkpoint_files_sha256"]
    spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": expected}}
    device = torch.device(args.device)
    vision, identity = load_frozen_vision(args.unit_checkpoint, spec, device)
    if identity["trainable_parameters"] != 0 or vision.training:
        raise RuntimeError("Original UniT Vision is not frozen in eval mode")
    digest_before = parameter_digest(vision)
    z_v = np.empty((len(pairs["pair_id"]), 8, 32), dtype=np.float32)
    first_repeat = None
    first_repeat_pass = None
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(z_v), args.batch_size):
            stop = min(start + args.batch_size, len(z_v))
            value = encode_batch(
                vision,
                pairs["current_frame"][start:stop],
                pairs["future_frame"][start:stop],
                executor,
            )
            if value.shape != (stop - start, 8, 32) or not np.isfinite(value).all():
                raise RuntimeError("Vision pilot output contract failure")
            z_v[start:stop] = value
            if start == 0:
                repeated = encode_batch(
                    vision,
                    pairs["current_frame"][start:stop],
                    pairs["future_frame"][start:stop],
                    executor,
                )
                first_repeat = float(np.max(np.abs(value - repeated)))
                first_repeat_pass = bool(np.allclose(value, repeated, atol=3e-3, rtol=3e-3))
            if stop % 1024 < args.batch_size or stop == len(z_v):
                print(json.dumps({"vision_pairs": stop, "total": len(z_v)}), flush=True)
        sample = min(8, len(z_v))
        order = np.arange(sample)[::-1]
        reordered = encode_batch(
            vision,
            pairs["current_frame"][:sample][order],
            pairs["future_frame"][:sample][order],
            executor,
        )[::-1]
        batch_one = np.concatenate(
            [
                encode_batch(
                    vision,
                    pairs["current_frame"][index : index + 1],
                    pairs["future_frame"][index : index + 1],
                    executor,
                )
                for index in range(sample)
            ]
        )
    digest_after = parameter_digest(vision)
    stability = {
        "tolerance": {"atol": 3e-3, "rtol": 3e-3},
        "repeated_extraction_max_abs": first_repeat,
        "sample_order_change_max_abs": float(np.max(np.abs(z_v[:sample] - reordered))),
        "batch_size_change_max_abs": float(np.max(np.abs(z_v[:sample] - batch_one))),
        "repeated_extraction_pass": first_repeat_pass,
        "sample_order_change_pass": bool(
            np.allclose(z_v[:sample], reordered, atol=3e-3, rtol=3e-3)
        ),
        "batch_size_change_pass": bool(
            np.allclose(z_v[:sample], batch_one, atol=3e-3, rtol=3e-3)
        ),
    }
    deterministic = all(
        stability[key]
        for key in (
            "repeated_extraction_pass",
            "sample_order_change_pass",
            "batch_size_change_pass",
        )
    )
    vision_result = {
        "schema": "tactile3d-unit.s4-2ds-vision-pilot.v1",
        "stage": "DS4_VISION",
        "role": "offline transition teacher; not formal S4.2-4",
        "frames": ["I_t", "I_t+27"],
        "latent_shape": [8, 32],
        "checkpoint_file_sha256": identity["original_unit_tokenizer_files_sha256"],
        "loading": identity["loading"],
        "eval_mode": not vision.training,
        "trainable_parameters": identity["trainable_parameters"],
        "parameter_digest_before": digest_before,
        "parameter_digest_after": digest_after,
        "parameter_unchanged": digest_before == digest_after,
        "finite": bool(np.isfinite(z_v).all()),
        "stability": stability,
        "deterministic": deterministic,
        "overall": "PASS"
        if deterministic and digest_before == digest_after and np.isfinite(z_v).all()
        else "FAIL",
        "failure_decision": None,
        "formal_test_loaded": False,
    }
    if vision_result["overall"] != "PASS":
        vision_result["failure_decision"] = "S4_2DS_VISION_PILOT_FAIL"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_ROOT / "pilot_pairs.npz",
        pair_id=pairs["pair_id"],
        episode_id=pairs["episode_id"],
        task=pairs["task"],
        source_trajectory_id=pairs["source_trajectory_id"],
        anchor_step=pairs["anchor_step"],
        future_step=pairs["future_step"],
        z_v=z_v,
        z_a=action["z_a"],
        z_c2=contact_codes["c2"],
        z_c3=contact_codes["c3"],
        ds_train_indices=action["ds_train_indices"],
        ds_dev_indices=action["ds_dev_indices"],
        contact_transition=contact["contact_transition"],
        force_trend=contact["force_trend"],
        dynamic=contact["dynamic"],
        h_current=contact["h_current"],
        h_future=contact["h_future"],
    )
    pair_manifest = {
        "schema": "tactile3d-unit.s4-2ds-pilot-pair-manifest.v1",
        "pairs": len(z_v),
        "ds_train_pairs": len(action["ds_train_indices"]),
        "ds_dev_pairs": len(action["ds_dev_indices"]),
        "join": ["pair_id", "episode_id", "task", "source_trajectory_id", "anchor t", "future t+27"],
        "pair_identity_exact": True,
        "future_offset_exact": bool(np.all(pairs["future_step"] - pairs["anchor_step"] == 27)),
        "source_group_exact": True,
        "shapes": {"z_v": [8, 32], "z_a": [8, 32], "z_c2": [8, 32], "z_c3": [8, 32]},
        "cache": ".local/cache/simulation/s4_2ds/pilot_pairs.npz",
        "cache_sha256": sha256_file(CACHE_ROOT / "pilot_pairs.npz"),
        "formal_validation_excluded": True,
        "formal_test_excluded": True,
        "formal_test_model_metrics_loaded": False,
    }
    atomic_json(ARTIFACT_ROOT / "vision_pilot.json", vision_result)
    atomic_json(ARTIFACT_ROOT / "pilot_pair_manifest.json", pair_manifest)
    print(json.dumps({"vision": vision_result["overall"], "pairs": len(z_v), "stability": stability}, indent=2))
    if vision_result["overall"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
