#!/usr/bin/env python3
"""Cold-recompute deterministic C1 V/A/C samples against the frozen cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    TReXPairedDataset,
    decode_rgb_frame,
    preprocess_trex_rgb,
)
from gr00t.tactile_unit.trex_action_bootstrap import TREX_EMBODIMENT_ID, ReleasedTokenizerSource  # noqa: E402
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import PUBLIC_TO_SOURCE, load_split  # noqa: E402
from scripts.tactile_unit.continuous_contact_bridge_common import load_frozen_vision, load_s2_model  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".local/cache/tactile_unit/vac_c1")
    parser.add_argument("--transition-cache", type=Path, default=ROOT / ".local/cache/contact_dynamics/s2_transition_pairs")
    parser.add_argument("--action-checkpoint", type=Path, default=ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt")
    parser.add_argument("--s2-checkpoint", type=Path, default=ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt")
    parser.add_argument("--artifact", type=Path, default=ROOT / ".local/artifacts/tactile_unit/vac_c1/cold_recompute.json")
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    return parser.parse_args()


def selected(pair_ids: np.ndarray, count: int) -> np.ndarray:
    order = sorted(
        range(len(pair_ids)),
        key=lambda index: hashlib.sha256(f"vac-c1-cold:{pair_ids[index]}".encode()).digest(),
    )
    return np.asarray(sorted(order[:count]), dtype=np.int64)


def compare(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float = 2e-5,
    rtol: float = 1e-5,
) -> dict[str, object]:
    difference = np.abs(np.asarray(expected, dtype=np.float64) - np.asarray(actual, dtype=np.float64))
    return {
        "exact": bool(np.array_equal(expected, actual)),
        "allclose": bool(np.allclose(expected, actual, atol=atol, rtol=rtol)),
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference": float(difference.max(initial=0.0)),
    }


def main():
    args = parse_args()
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required")
    spec = json.loads(args.config.read_text())
    device, lock_handle, gpu = resolve_device(args.device)
    try:
        vision_spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": spec["frozen_identity"]["original_unit_tokenizer_files_sha256"]}}
        vision, vision_identity = load_frozen_vision(args.unit_checkpoint, vision_spec, device)
        s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
        source = ReleasedTokenizerSource.open(args.unit_checkpoint / "tokenizer")
        action, _ = load_shared_transition_checkpoint(args.action_checkpoint, source)
        action.eval().requires_grad_(False).to(device)
        before = {"action": parameter_digest(action), "s2_encoder": parameter_digest(s2.encoder), "s2_decoder": parameter_digest(s2.decoder)}
        result = {"schema": "tactile3d-unit.vac-c1-cold-recompute.v1", "gpu": gpu, "vision_identity": vision_identity, "splits": {}}
        with torch.inference_mode():
            for public, source_split in PUBLIC_TO_SOURCE.items():
                cache = load_split(args.cache_root, public, verify_hashes=False)
                indices = selected(np.asarray(cache.arrays["pair_id"]), args.samples_per_split)
                dataset = TReXPairedDataset(args.dataset_root, args.transition_cache, split=source_split)
                rows = []
                for row in indices:
                    source_index = int(cache.arrays["source_index"][row])
                    record = dataset.record(source_index)
                    path = args.dataset_root / record["vision"]["relative_path"]
                    obs = preprocess_trex_rgb(decode_rgb_frame(path, record["vision"]["current"]["packed_timestamp"]))
                    goal = preprocess_trex_rgb(decode_rgb_frame(path, record["vision"]["future"]["packed_timestamp"]))
                    obs_t = torch.from_numpy(obs)[None, None].to(device, dtype=vision.dtype)
                    goal_t = torch.from_numpy(goal)[None, None].to(device, dtype=vision.dtype)
                    values, _, _ = vision.vision_branch(obs_t, goal_t, batch_size=1)
                    z_v = vision.vq_down_resampler(values).float().cpu().numpy()[0]
                    state = torch.from_numpy(np.array(cache.arrays["state"][row:row + 1], copy=True)).to(device)
                    action_chunk = torch.from_numpy(np.array(cache.arrays["action"][row:row + 1], copy=True)).to(device)
                    embodiment = torch.full((1,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
                    z_a = action.encode(state, action_chunk, embodiment)[0].float().cpu().numpy()[0]
                    current = torch.from_numpy(np.array(cache.arrays["h_current"][row:row + 1], copy=True)).to(device)
                    future = torch.from_numpy(np.array(cache.arrays["h_future"][row:row + 1], copy=True)).to(device)
                    z_c = s2.encoder(current, future).float().cpu().numpy()[0]
                    checks = {
                        # The frozen Vision tower runs in float16. Its GEMM reduction
                        # order changes between the extraction batch and this cold
                        # single-sample path, so use an explicit fp16-scale tolerance.
                        "vision": compare(
                            np.asarray(cache.arrays["z_v"][row]), z_v, atol=3e-3, rtol=3e-3
                        ),
                        "action": compare(np.asarray(cache.arrays["z_a"][row]), z_a),
                        "contact": compare(np.asarray(cache.arrays["z_c"][row]), z_c),
                    }
                    if not all(value["allclose"] for value in checks.values()):
                        raise RuntimeError(
                            "cold recompute mismatch for "
                            f"{cache.arrays['pair_id'][row]}: {json.dumps(checks, sort_keys=True)}"
                        )
                    rows.append({"pair_id": str(cache.arrays["pair_id"][row]), "checks": checks})
                result["splits"][public] = rows
        after = {"action": parameter_digest(action), "s2_encoder": parameter_digest(s2.encoder), "s2_decoder": parameter_digest(s2.decoder)}
        result["native_identity_before"] = before
        result["native_identity_after"] = after
        result["status"] = "PASS" if before == after else "FAIL"
        if result["status"] != "PASS":
            raise RuntimeError("native parameter identity changed during cold recompute")
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
