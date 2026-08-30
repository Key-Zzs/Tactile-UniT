#!/usr/bin/env python3
"""Audit provenance and build exact C3-MS-CC-R Action perturbation caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.c3msccr_exact_action_closure import (  # noqa: E402
    ACTION_ORDERING,
    VARIANTS,
    canonical_action_from_raw,
    canonical_json_sha256,
    deterministic_temporal_orders,
    raw_action_from_canonical,
    row_mse,
    same_split_different_indices,
    sha256_file,
)
from gr00t.tactile_unit.continuous_vac_shared_space import load_checkpoint  # noqa: E402
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID, ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from scripts.tactile_unit.c3mscc_runtime import atomic_json  # noqa: E402
from scripts.tactile_unit.vac_runtime_common import resolve_device  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c3msccr_exact_action_closure.json"
SPLIT_OFFSET = {"train": 0, "validation": 100_000, "test": 200_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("audit", "build", "all"), default="all")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def tokenizer_root(args: argparse.Namespace) -> Path:
    if args.unit_checkpoint is None:
        raise RuntimeError("C3MSCCR_EXACT_ACTION_EVIDENCE_UNAVAILABLE: UNIT_FULLDATA_CKPT is required")
    root = args.unit_checkpoint / "tokenizer"
    if not root.is_dir():
        raise RuntimeError("C3MSCCR_EXACT_ACTION_EVIDENCE_UNAVAILABLE: tokenizer directory missing")
    return root


def load_action_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / config["runtime"]["action_checkpoint"]
    return torch.load(path, map_location="cpu", weights_only=False)


def provenance_audit(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    accepted = config["accepted"]
    runtime = config["runtime"]
    root = tokenizer_root(args)
    files = accepted["original_unit_tokenizer_files_sha256"]
    actual_files = {name: sha256_file(root / name) for name in files}
    action_path = ROOT / runtime["action_checkpoint"]
    payload = load_action_payload(config)
    source = ReleasedTokenizerSource.open(root)
    feature_stats = payload.get("feature_stats", {})
    paths = {
        "c3mscc": ROOT / runtime["c3mscc_root"] / "selected.pt",
        "action": action_path,
        "c1": ROOT / runtime["c1_cache_root"] / "manifest.json",
        "c2r": ROOT / runtime["c2r_checkpoint"],
        "t0": ROOT / runtime["c3mscc_root"] / "trial_00_T0/best.pt",
        "t1": ROOT / runtime["c3mscc_root"] / "trial_01_T1/best.pt",
    }
    expected_paths = {
        "c3mscc": accepted["c3mscc_checkpoint_sha256"],
        "action": accepted["action_checkpoint_sha256"],
        "c1": accepted["c1_manifest_sha256"],
        "c2r": accepted["c2r_checkpoint_sha256"],
        "t0": accepted["t0_checkpoint_sha256"],
        "t1": accepted["t1_checkpoint_sha256"],
    }
    actual_paths = {name: sha256_file(path) for name, path in paths.items()}
    checks = {
        "base_files": actual_files == files,
        "frozen_paths": actual_paths == expected_paths,
        "checkpoint_schema": payload.get("schema") == "tactile3d-unit.s3-3-r-shared-transition.v1",
        "candidate": payload.get("candidate") == "R1-P",
        "source_identity": payload.get("source_identity") == source.identity,
        "feature_stats": feature_stats.get("canonical_sha256") == accepted["action_feature_stats_canonical_sha256"],
        "feature_stats_train_only": feature_stats.get("fit_split") == "frozen train split only",
        "old_rows": source.old_rows_digest() == accepted["old_action_rows_digest"],
        "embodiment": int(config["contract"]["embodiment_id"]) == TREX_EMBODIMENT_ID,
        "ordering": tuple(config["contract"]["ordering"]) == ACTION_ORDERING,
    }
    decision = "PASS" if all(checks.values()) else "C3MSCCR_ACTION_PROVENANCE_INVALID"
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-action-provenance.v1",
        "decision": decision,
        "checks": checks,
        "base_file_hashes": actual_files,
        "frozen_path_hashes": actual_paths,
        "source_identity": source.identity,
        "feature_stats_canonical_sha256": feature_stats.get("canonical_sha256"),
        "feature_stats_fit_split": feature_stats.get("fit_split"),
        "old_action_rows_digest": source.old_rows_digest(),
        "embodiment_id": TREX_EMBODIMENT_ID,
        "category_capacity": 32,
        "normalization_refit": False,
        "transition_stats_refit": False,
        "exact_inference_capability": all(checks.values()),
    }
    artifact = ROOT / runtime["artifact_root"] / "action_provenance_audit.json"
    atomic_json(artifact, result)
    if decision != "PASS":
        raise RuntimeError(decision)
    return result


def create_array(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def build_split(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    runtime = config["runtime"]
    source_root = ROOT / runtime["c1_cache_root"] / split
    output_root = ROOT / runtime["exact_cache_root"] / split
    output_root.mkdir(parents=True, exist_ok=True)
    state = np.load(source_root / "state.npy", mmap_mode="r", allow_pickle=False)
    action = np.load(source_root / "action.npy", mmap_mode="r", allow_pickle=False)
    episode = np.load(source_root / "episode_id.npy", mmap_mode="r", allow_pickle=False)
    pair_id = np.load(source_root / "pair_id.npy", mmap_mode="r", allow_pickle=False)
    source_index = np.load(source_root / "source_index.npy", mmap_mode="r", allow_pickle=False)
    dynamic = np.load(source_root / "dynamic.npy", mmap_mode="r", allow_pickle=False)
    count = len(state)
    if action.shape != (count, 16, 128) or state.shape != (count, 128):
        raise RuntimeError("STRUCTURAL_FAIL: C1 Action/state geometry changed")
    split_seed = int(config["perturbations"]["shuffle_seed"]) + SPLIT_OFFSET[split]
    different_seed = int(config["perturbations"]["different_episode_seed"]) + SPLIT_OFFSET[split]
    orders = deterministic_temporal_orders(count, split_seed)
    different = same_split_different_indices(episode, different_seed)
    np.save(output_root / "pair_id.npy", np.asarray(pair_id), allow_pickle=False)
    np.save(output_root / "source_index.npy", np.asarray(source_index), allow_pickle=False)
    np.save(output_root / "shuffle_order.npy", orders, allow_pickle=False)
    np.save(output_root / "different_source_index.npy", different.astype(np.int32), allow_pickle=False)

    source = ReleasedTokenizerSource.open(tokenizer_root(args))
    action_model, _ = load_shared_transition_checkpoint(
        ROOT / runtime["action_checkpoint"], source, map_location=device
    )
    shared, _ = load_checkpoint(ROOT / runtime["c2r_checkpoint"], map_location=device)
    action_model.eval().requires_grad_(False).to(device)
    shared.eval().requires_grad_(False).to(device)
    feature_stats = load_action_payload(config)["feature_stats"]
    arrays: dict[str, np.memmap] = {}
    required_paths = []
    for variant in VARIANTS:
        required_paths.extend(
            (output_root / f"z_a_{variant}.npy", output_root / f"u_a_{variant}.npy")
        )
        if split == config["cache"]["decoder_integrity_split"]:
            required_paths.append(output_root / f"decoder_mse_{variant}.npy")
    reuse_complete = all(path.is_file() for path in required_paths)
    for variant in VARIANTS:
        if not reuse_complete:
            arrays[f"z_a_{variant}"] = create_array(
                output_root / f"z_a_{variant}.npy", np.dtype("float32"), (count, 8, 32)
            )
            arrays[f"u_a_{variant}"] = create_array(
                output_root / f"u_a_{variant}.npy", np.dtype("float32"), (count, 8, 32)
            )
        if split == config["cache"]["decoder_integrity_split"]:
            if not reuse_complete:
                arrays[f"decoder_mse_{variant}"] = create_array(
                    output_root / f"decoder_mse_{variant}.npy", np.dtype("float32"), (count,)
                )
    batch_size = int(args.batch_size or config["cache"]["batch_size"])
    started = time.monotonic()
    with torch.inference_mode():
        if not reuse_complete:
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                state_np = np.array(state[start:stop], copy=True)
                raw_current = raw_action_from_canonical(
                    np.array(action[start:stop], copy=True), feature_stats
                )
                raw_different = raw_action_from_canonical(
                    np.array(action[different[start:stop]], copy=True), feature_stats
                )
                order = orders[start:stop].astype(np.int64)
                row = np.arange(stop - start, dtype=np.int64)[:, None]
                raw_variants = {
                    "correct": raw_current,
                    "reversed": raw_current[:, ::-1].copy(),
                    "shuffled": raw_current[row, order].copy(),
                    "different": raw_different,
                }
                state_tensor = torch.from_numpy(state_np).to(device)
                embodiment = torch.full(
                    (stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device
                )
                correct_target = torch.from_numpy(
                    np.array(action[start:stop], copy=True)
                ).to(device)
                for variant in VARIANTS:
                    canonical = canonical_action_from_raw(raw_variants[variant], feature_stats)
                    if np.any(canonical[..., 58:] != 0):
                        raise AssertionError("Action padding changed after perturbation")
                    candidate = torch.from_numpy(canonical).to(device)
                    z_a, state_features, _ = action_model.encode(
                        state_tensor, candidate, embodiment
                    )
                    u_a = shared.encode("action", z_a)
                    arrays[f"z_a_{variant}"][start:stop] = z_a.float().cpu().numpy()
                    arrays[f"u_a_{variant}"][start:stop] = u_a.float().cpu().numpy()
                    decoder_key = f"decoder_mse_{variant}"
                    if decoder_key in arrays:
                        decoded = action_model.decode(z_a, state_features, embodiment)
                        arrays[decoder_key][start:stop] = torch.square(
                            decoded - correct_target
                        ).flatten(1).mean(1).float().cpu().numpy()
    for value in arrays.values():
        value.flush()
    expected_z = np.load(source_root / "z_a.npy", mmap_mode="r", allow_pickle=False)
    expected_u = np.load(
        ROOT / runtime["c3dp_cache_root"] / split / "u_a.npy", mmap_mode="r", allow_pickle=False
    )
    actual_z = np.load(output_root / "z_a_correct.npy", mmap_mode="r", allow_pickle=False)
    actual_u = np.load(output_root / "u_a_correct.npy", mmap_mode="r", allow_pickle=False)
    tolerance = float(config["cache"]["correct_reproduction_atol"])
    z_max = float(np.max(np.abs(np.asarray(actual_z) - np.asarray(expected_z))))
    u_max = float(np.max(np.abs(np.asarray(actual_u) - np.asarray(expected_u))))
    if z_max > tolerance or u_max > tolerance:
        raise RuntimeError("C3MSCCR_AR_TEMPORAL_PROVENANCE_FAIL: correct latent reproduction")
    arrays_meta = {}
    for path in sorted(output_root.glob("*.npy")):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        arrays_meta[path.name] = {
            "shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_file(path)
        }
    return {
        "split": split,
        "count": count,
        "dynamic": int(np.asarray(dynamic).sum()),
        "shuffle_seed": split_seed,
        "different_episode_seed": different_seed,
        "same_split_different_episode": bool(np.all(np.asarray(episode)[different] != np.asarray(episode))),
        "correct_reproduction": {
            "tolerance": tolerance, "z_a_max_abs": z_max, "u_a_max_abs": u_max,
            "pass": z_max <= tolerance and u_max <= tolerance,
        },
        "seconds": time.monotonic() - started,
        "reused_complete_cache": reuse_complete,
        "arrays": arrays_meta,
    }


def build(args: argparse.Namespace, config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    splits = [args.split] if args.split else list(config["cache"]["splits"])
    result = {
        "schema": "tactile3d-unit.vac-c3msccr-exact-action-cache.v1",
        "perturbation_order": "raw action -> perturb -> accepted normalization -> transition features -> R1-P -> frozen A-R -> frozen P_a",
        "reverse_before_feature_construction": True,
        "state_relative_recomputed": True,
        "first_difference_recomputed": True,
        "action_interval": "a_t:t+15",
        "no_a_t_plus_16": True,
        "embodiment_id": TREX_EMBODIMENT_ID,
        "rq": False,
        "splits": {},
    }
    artifact_root = ROOT / config["runtime"]["artifact_root"]
    previous = artifact_root / "exact_action_cache_manifest.json"
    if previous.is_file():
        result["splits"].update(json.loads(previous.read_text()).get("splits", {}))
    for split in splits:
        result["splits"][split] = build_split(args, config, split, device)
        atomic_json(previous, result)
    result["canonical_sha256"] = canonical_json_sha256(result)
    atomic_json(previous, result)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device == "cpu":
        torch.set_num_threads(int(config["cache"]["cpu_threads"]))
    device, lock_handle, gpu = resolve_device(
        args.device, allowed_physical=tuple(str(value) for value in config["gpu"]["allowed_physical"])
    )
    try:
        if args.phase in {"audit", "all"}:
            provenance_audit(args, config)
        if args.phase in {"build", "all"}:
            build(args, config, device)
        print(json.dumps({"status": "PASS", "device": str(device), "gpu": gpu}, sort_keys=True))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
