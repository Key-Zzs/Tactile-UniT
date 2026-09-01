#!/usr/bin/env python3
"""Build exact train/validation Contact-State transition caches for S4.2-3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_2_dataset import sha256_file  # noqa: E402
from gr00t.simulation.s4_2_normalization import TactileNormalization  # noqa: E402
from gr00t.simulation.sim_contact_models import load_teacher_checkpoint  # noqa: E402

PAIR_ROOT = ROOT / ".local/cache/simulation/s4_2/pairs"
CACHE_ROOT = ROOT / ".local/cache/simulation/s4_2r/contact_dynamics"
ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_2r"
ACCEPTANCE = ROOT / "configs/simulation/s4_2r_contact_state_acceptance.json"
ORIGINAL_EVALUATION = ROOT / ".local/artifacts/simulation/s4_2/teacher_evaluation.json"
SPLITS = ("train", "validation")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def encode(
    model: torch.nn.Module,
    history: np.ndarray,
    normalizer: TactileNormalization,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = normalizer.transform(history).astype(np.float32)
    output = np.empty((len(values), 256), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            batch = torch.from_numpy(values[start:stop]).to(device)
            output[start:stop] = model.encode(batch).cpu().numpy()
    return output


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, default=PAIR_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    if acceptance["final_state"] != "S4_2R_CONTACT_STATE_ACCEPTED_EXISTING":
        raise RuntimeError("S4.2-3 requires accepted Contact-State")
    if acceptance["test_loaded"] is not False or acceptance["s4_2_3_allowed"] is not True:
        raise RuntimeError("Contact-State acceptance does not authorize S4.2-3")
    checkpoint_path = ROOT / acceptance["accepted_teacher"]["checkpoint"]
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != acceptance["accepted_teacher"]["checkpoint_sha256"]:
        raise RuntimeError("accepted Contact-State checkpoint identity mismatch")
    teacher, checkpoint = load_teacher_checkpoint(checkpoint_path, "cpu")
    teacher.requires_grad_(False).eval()
    device = torch.device(args.device)
    teacher = teacher.to(device)
    digest_before = parameter_digest(teacher)
    normalizer = TactileNormalization.from_json(checkpoint["normalization"])
    raw = {split: load_npz(args.pair_root / f"{split}.npz") for split in SPLITS}
    dynamic_threshold = float(np.quantile(raw["train"]["force_delta_abs"], 0.70))
    original = json.loads(ORIGINAL_EVALUATION.read_text(encoding="utf-8"))
    force_deadband = float(original["probes"]["trend_deadband_newton"])
    args.cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "tactile3d-unit.s4-2r-contact-dynamics-cache.v1",
        "teacher_checkpoint_sha256": checkpoint_sha,
        "normalization": normalizer.to_json(),
        "window_contract": {
            "current_relative_steps": [-25, 0],
            "future_relative_steps": [2, 27],
            "current_samples": 26,
            "future_samples": 26,
            "anchor_offset_steps": 27,
            "anchor_delta_sec": 0.54,
            "raw_overlap": False,
        },
        "dynamic_rule": "force_delta_abs > TRAIN q70",
        "dynamic_q70_threshold_newton": dynamic_threshold,
        "force_trend_deadband_newton": force_deadband,
        "splits": {},
        "test_cached": False,
        "test_loaded": False,
    }
    for split in SPLITS:
        values = raw[split]
        h_current = encode(
            teacher, values["current_history"], normalizer, device, args.batch_size
        )
        h_future = encode(
            teacher, values["future_history"], normalizer, device, args.batch_size
        )
        current = values["current_history"][:, -1].reshape(-1, 5, 6)
        future = values["future_history"][:, -1].reshape(-1, 5, 6)
        current_contact = current[:, :, 0] > 0.5
        future_contact = future[:, :, 0] > 0.5
        region_change = (
            future_contact.astype(np.int8) - current_contact.astype(np.int8) + 1
        )
        force_delta = values["future_total_force"] - values["current_total_force"]
        force_trend = np.ones(len(force_delta), dtype=np.int8)
        force_trend[force_delta < -force_deadband] = 0
        force_trend[force_delta > force_deadband] = 2
        dynamic = values["force_delta_abs"] > dynamic_threshold
        destination = args.cache_root / f"{split}.npz"
        np.savez_compressed(
            destination,
            pair_id=values["pair_id"],
            episode_id=values["episode_id"],
            task=values["task"],
            source_trajectory_id=values["source_trajectory_id"],
            anchor_step=values["anchor_step"],
            future_step=values["future_step"],
            h_current=h_current,
            h_future=h_future,
            contact_transition=values["contact_transition"],
            current_total_force=values["current_total_force"],
            future_total_force=values["future_total_force"],
            force_delta=force_delta.astype(np.float32),
            force_delta_abs=values["force_delta_abs"],
            force_trend=force_trend,
            current_region_contact=current_contact,
            future_region_contact=future_contact,
            region_contact_change=region_change,
            dynamic=dynamic,
        )
        manifest["splits"][split] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256_file(destination),
            "pairs": int(len(h_current)),
            "h_current_shape": list(h_current.shape),
            "h_future_shape": list(h_future.shape),
            "dynamic_pairs": int(dynamic.sum()),
            "contact_regimes": {
                str(code): int(np.sum(values["contact_transition"] == code))
                for code in range(4)
            },
        }
    digest_after = parameter_digest(teacher)
    manifest["teacher_parameter_digest_before"] = digest_before
    manifest["teacher_parameter_digest_after"] = digest_after
    manifest["teacher_unchanged"] = digest_before == digest_after
    manifest["teacher_requires_grad"] = any(
        parameter.requires_grad for parameter in teacher.parameters()
    )
    if not manifest["teacher_unchanged"] or manifest["teacher_requires_grad"]:
        raise RuntimeError("accepted Contact-State teacher was not frozen")
    atomic_json(args.artifacts / "contact_dynamics_cache_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
