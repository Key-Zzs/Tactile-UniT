#!/usr/bin/env python3
"""Freeze and extract the C1 three-modal paired latent dataset."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.tactile_unit.compatibility import parameter_digest  # noqa: E402
from gr00t.tactile_unit.paired_contract import (  # noqa: E402
    CANONICAL_FPS,
    TReXPairedDataset,
    data_relative_path,
    discover_dataset_revision,
    load_episode_video_pointers,
    load_info,
    normalize_and_pad_trex_state_action,
    preprocess_trex_rgb,
    sha256_file,
    sha256_json,
)
from gr00t.tactile_unit.trex_action_bootstrap import (  # noqa: E402
    TREX_EMBODIMENT_ID,
    ReleasedTokenizerSource,
)
from gr00t.tactile_unit.trex_action_transition import load_shared_transition_checkpoint  # noqa: E402
from gr00t.tactile_unit.vac_latent_dataset import (  # noqa: E402
    CACHE_SCHEMA,
    PUBLIC_TO_SOURCE,
    REQUIRED_ARRAYS,
    atomic_json,
    canonical_pair_ids,
    deterministic_train_subset,
    deterministic_uniform_subset,
    pair_id_digest,
    split_manifest,
    write_npy_atomic,
)
from scripts.tactile_unit.continuous_contact_bridge_common import (  # noqa: E402
    load_frozen_vision,
    load_s2_model,
)
from scripts.tactile_unit.vac_runtime_common import resolve_device, set_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/tactile_unit/c1_vac_latent_dataset.json"
DEFAULT_TRANSITIONS = ROOT / ".local/cache/contact_dynamics/s2_transition_pairs"
DEFAULT_CONTACT_CODES = ROOT / ".local/cache/contact_dynamics/s2_codes"
DEFAULT_PAIRED_MANIFEST = ROOT / ".local/artifacts/tactile_unit/s3_1/paired_eval_manifest.json"
DEFAULT_INTEGRATION_CACHE = ROOT / ".local/cache/tactile_unit/integration/canonical_vac_teachers.npz"
DEFAULT_STATS = ROOT / ".local/artifacts/tactile_unit/s3_1/state_action_normalization.json"
DEFAULT_ACTION = ROOT / ".local/experiments/tactile_unit/s3_3_r/selected.pt"
DEFAULT_S2 = ROOT / ".local/experiments/contact_dynamics/s2_models/proposed_best.pt"
DEFAULT_S1 = ROOT / ".local/experiments/tactile_teacher/s1_teacher/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("selection", "contact", "action", "vision", "finalize", "all"), default="all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=Path(os.environ.get("TREX_DATASET_DIR", ROOT / ".local/datasets/tactile_teacher/trex_dataset")))
    parser.add_argument("--unit-checkpoint", type=Path, default=Path(os.environ["UNIT_FULLDATA_CKPT"]) if os.environ.get("UNIT_FULLDATA_CKPT") else None)
    parser.add_argument("--transition-cache", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--contact-codes", type=Path, default=DEFAULT_CONTACT_CODES)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--integration-cache", type=Path, default=DEFAULT_INTEGRATION_CACHE)
    parser.add_argument("--state-action-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--action-checkpoint", type=Path, default=DEFAULT_ACTION)
    parser.add_argument("--s2-checkpoint", type=Path, default=DEFAULT_S2)
    parser.add_argument("--s1-checkpoint", type=Path, default=DEFAULT_S1)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--vision-cluster-gap", type=float, default=2.0)
    parser.add_argument("--only-split", choices=("train", "validation", "test"))
    return parser.parse_args()


def source_array(root: Path, split: str, name: str) -> np.ndarray:
    return np.load(root / split / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def verify_sources(args: argparse.Namespace, spec: Mapping[str, Any]) -> dict[str, Any]:
    identity = spec["frozen_identity"]
    actual = {
        "paired_manifest_sha256": sha256_file(args.paired_manifest),
        "s2_transition_manifest_sha256": sha256_file(args.transition_cache / "manifest.json"),
        "action_checkpoint_sha256": sha256_file(args.action_checkpoint),
        "s2_checkpoint_sha256": sha256_file(args.s2_checkpoint),
        "s1_teacher_checkpoint_sha256": sha256_file(args.s1_checkpoint),
    }
    for name, digest in actual.items():
        if digest != identity[name]:
            raise RuntimeError(f"frozen source identity mismatch: {name}")
    manifest = json.loads(args.paired_manifest.read_text())
    if manifest.get("canonical_sha256") != identity["paired_manifest_canonical_sha256"]:
        raise RuntimeError("canonical 960 manifest content digest mismatch")
    ids = [str(row["pair_id"]) for row in manifest["rows"]]
    if len(ids) != 960 or len(set(ids)) != 960 or pair_id_digest(ids) != identity["canonical_pair_id_digest"]:
        raise RuntimeError("canonical 960 pair identity changed")
    revision = discover_dataset_revision(args.dataset_root)
    if revision != "bf0eb24c4b8bdd95752b553f0fc50e46a22f1cc8":
        raise RuntimeError("T-Rex dataset revision mismatch")
    if args.unit_checkpoint is not None:
        tokenizer = args.unit_checkpoint / "tokenizer"
        token_hashes = {
            name: sha256_file(tokenizer / name)
            for name in identity["original_unit_tokenizer_files_sha256"]
        }
        if token_hashes != identity["original_unit_tokenizer_files_sha256"]:
            raise RuntimeError("Original UniT Vision checkpoint identity mismatch")
        actual["original_unit_tokenizer_files_sha256"] = token_hashes
    actual["contact_code_cache_sha256"] = {
        split: sha256_file(args.contact_codes / f"{split}.npy")
        for split in ("train", "val", "test")
        if (args.contact_codes / f"{split}.npy").is_file()
    }
    return {**actual, "dataset_revision": revision, "canonical_960_pair_id_digest": pair_id_digest(ids)}


def worktree_paths() -> list[Path]:
    output = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=ROOT,
        text=True, capture_output=True, check=True,
    ).stdout
    return [Path(line.split(" ", 1)[1]) for line in output.splitlines() if line.startswith("worktree ")]


def reuse_c0_rows(split: str) -> tuple[set[int], list[Path]]:
    rows: set[int] = set()
    found: list[Path] = []
    for worktree in worktree_paths():
        path = worktree / ".local/cache/tactile_unit/c0" / f"paired_{split}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            if "source_index" not in payload.files or "pair_id" not in payload.files:
                continue
            rows.update(map(int, payload["source_index"]))
        found.append(path)
    return rows, found


def create_memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def freeze_selection(args: argparse.Namespace, spec: Mapping[str, Any], cache_root: Path) -> dict[str, Any]:
    selection_path = cache_root / "selection_manifest.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        body = {key: value for key, value in selection.items() if key != "canonical_sha256"}
        if sha256_json(body) != selection.get("canonical_sha256"):
            raise RuntimeError("existing C1 selection manifest is corrupt")
        return selection

    seed = int(spec["seed"])
    selected: dict[str, np.ndarray] = {}
    reuse_paths: dict[str, list[str]] = {}
    for public, source in PUBLIC_TO_SOURCE.items():
        episode = source_array(args.transition_cache, source, "episode_id")
        expected_full = int(spec["full_legal_pairs"][public])
        if len(episode) != expected_full:
            raise RuntimeError(f"{public} full legal pair count changed")
        count = int(spec["cached_pairs"][public])
        forced, paths = reuse_c0_rows(public)
        reuse_paths[public] = [f"worktree-local:{path.name}" for path in paths]
        if public == "train":
            indices = deterministic_train_subset(
                count=count,
                seed=seed,
                primitive_id=source_array(args.transition_cache, source, "primitive_id"),
                object_id=source_array(args.transition_cache, source, "object_id"),
                dynamic=source_array(args.transition_cache, source, "dynamic"),
                contact_transition=source_array(args.transition_cache, source, "contact_transition"),
                forced_indices=forced,
            )
        elif public == "validation":
            indices = deterministic_uniform_subset(len(episode), count, seed=seed + 1, split=public)
        else:
            indices = np.arange(len(episode), dtype=np.int64)
        if len(indices) != count or len(np.unique(indices)) != count:
            raise RuntimeError(f"invalid frozen {public} subset")
        selected[public] = indices

    selection: dict[str, Any] = {
        "schema": "tactile3d-unit.vac-c1-selection.v1",
        "seed": seed,
        "frozen_before_training": True,
        "test_tuned": False,
        "splits": {},
        "reuse_sources": reuse_paths,
    }
    for public, indices in selected.items():
        source = PUBLIC_TO_SOURCE[public]
        split_root = cache_root / public
        split_root.mkdir(parents=True, exist_ok=True)
        episode = np.asarray(source_array(args.transition_cache, source, "episode_id")[indices], dtype=np.int32)
        anchor = np.asarray(source_array(args.transition_cache, source, "anchor_frame")[indices], dtype=np.int32)
        pair_ids = canonical_pair_ids(public, episode, anchor)
        scalar = {
            "pair_id": pair_ids,
            "episode_id": episode,
            "t": anchor,
            "t_future": anchor + 16,
            "source_index": indices.astype(np.int32),
            "dynamic": np.asarray(source_array(args.transition_cache, source, "dynamic")[indices], dtype=np.bool_),
            "contact_transition": np.asarray(source_array(args.transition_cache, source, "contact_transition")[indices], dtype=np.int8),
            "force_trend_class": np.asarray(source_array(args.transition_cache, source, "force_trend_class")[indices], dtype=np.int8),
            "primitive_id": np.asarray(source_array(args.transition_cache, source, "primitive_id")[indices], dtype=np.int16),
            "object_id": np.asarray(source_array(args.transition_cache, source, "object_id")[indices], dtype=np.int16),
            "task_id": np.asarray(source_array(args.transition_cache, source, "task_id")[indices], dtype=np.int64),
            "current_force": np.asarray(source_array(args.transition_cache, source, "current_force")[indices], dtype=np.float32),
            "future_force": np.asarray(source_array(args.transition_cache, source, "future_force")[indices], dtype=np.float32),
        }
        for name, value in scalar.items():
            write_npy_atomic(split_root / f"{name}.npy", value)
        for name, (dtype, trailing) in REQUIRED_ARRAYS.items():
            path = split_root / f"{name}.npy"
            if path.exists():
                continue
            array = create_memmap(path, dtype, (len(indices), *trailing))
            array[...] = 0
            array.flush()
        selection["splits"][public] = {
            "count": len(indices),
            "source_index_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
            "pair_id_digest": pair_id_digest(pair_ids),
            "source_order": "ascending S2 source_index",
        }
    body = dict(selection)
    selection["canonical_sha256"] = sha256_json(body)
    atomic_json(selection_path, selection)
    return selection


def phase_marker(cache_root: Path, phase: str, value: Mapping[str, Any]) -> None:
    atomic_json(cache_root / "phases" / f"{phase}.json", dict(value))


def contact_phase(args: argparse.Namespace, spec: Mapping[str, Any], cache_root: Path, device: torch.device) -> dict[str, Any]:
    s2 = load_s2_model(args.s2_checkpoint, device).eval().requires_grad_(False)
    identity = spec["frozen_identity"]
    if parameter_digest(s2.encoder) != identity["s2_encoder_parameter_digest"] or parameter_digest(s2.decoder) != identity["s2_decoder_parameter_digest"]:
        raise RuntimeError("S2 encoder/decoder identity mismatch")
    result: dict[str, Any] = {}
    with torch.inference_mode():
        for public, source in PUBLIC_TO_SOURCE.items():
            if args.only_split is not None and public != args.only_split:
                continue
            split_root = cache_root / public
            indices = np.load(split_root / "source_index.npy", mmap_mode="r")
            current = source_array(args.transition_cache, source, "current")
            future = source_array(args.transition_cache, source, "future")
            h_current = np.lib.format.open_memmap(split_root / "h_current.npy", mode="r+")
            h_future = np.lib.format.open_memmap(split_root / "h_future.npy", mode="r+")
            z_c = np.lib.format.open_memmap(split_root / "z_c.npy", mode="r+")
            h_current[...] = current[indices]
            h_future[...] = future[indices]
            codes_path = args.contact_codes / f"{source}.npy"
            reused = 0
            if codes_path.is_file():
                codes = np.load(codes_path, mmap_mode="r", allow_pickle=False)
                if codes.shape != (len(current), 8, 32) or not np.isfinite(codes).all():
                    raise RuntimeError(f"invalid accepted Contact cache for {public}")
                z_c[...] = codes[indices]
                reused = len(indices)
            else:
                for start in range(0, len(indices), args.batch_size):
                    stop = min(start + args.batch_size, len(indices))
                    current_batch = torch.from_numpy(np.asarray(h_current[start:stop])).to(device)
                    future_batch = torch.from_numpy(np.asarray(h_future[start:stop])).to(device)
                    z_c[start:stop] = s2.encoder(current_batch, future_batch).float().cpu().numpy()
            h_current.flush(); h_future.flush(); z_c.flush()
            result[public] = {"count": len(indices), "accepted_cache_rows_reused": reused}
    phase_marker(cache_root, "contact", result)
    return result


def load_normalization(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("mode") != "mean_std" or value.get("fit_split") != "frozen S1 train episodes only":
        raise RuntimeError("state/action normalization is not accepted train-only mean/std")
    return value


def extract_state_action(args: argparse.Namespace, cache_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    stats = load_normalization(args.state_action_stats)
    pointers = {item.episode_id: item for item in load_episode_video_pointers(args.dataset_root)}
    info = load_info(args.dataset_root)
    result: dict[str, Any] = {}
    for public in PUBLIC_TO_SOURCE:
        if args.only_split is not None and public != args.only_split:
            continue
        split_root = cache_root / public
        episode = np.load(split_root / "episode_id.npy", mmap_mode="r")
        anchor = np.load(split_root / "t.npy", mmap_mode="r")
        state_out = np.lib.format.open_memmap(split_root / "state.npy", mode="r+")
        action_out = np.lib.format.open_memmap(split_root / "action.npy", mode="r+")
        grouped: dict[str, list[int]] = defaultdict(list)
        for row, episode_id in enumerate(episode):
            pointer = pointers[int(episode_id)]
            grouped[data_relative_path(info, pointer)].append(row)
        for relative_path, rows in sorted(grouped.items()):
            table = pq.read_table(
                args.dataset_root / relative_path,
                columns=["index", "episode_index", "frame_index", "observation.state", "action"],
            )
            global_index = np.asarray(table["index"], dtype=np.int64)
            if len(global_index) == 0 or np.any(np.diff(global_index) != 1):
                raise RuntimeError(f"non-contiguous data file {relative_path}")
            first = int(global_index[0])
            episodes = np.asarray(table["episode_index"], dtype=np.int64)
            frames = np.asarray(table["frame_index"], dtype=np.int64)
            for row in rows:
                pointer = pointers[int(episode[row])]
                offset = pointer.dataset_from_index + int(anchor[row]) - first
                if int(episodes[offset]) != int(episode[row]) or int(frames[offset]) != int(anchor[row]):
                    raise RuntimeError("state/action pair identity mismatch")
                if not np.all(episodes[offset:offset + 16] == int(episode[row])) or not np.array_equal(frames[offset:offset + 16], np.arange(int(anchor[row]), int(anchor[row]) + 16)):
                    raise RuntimeError("action window crosses canonical episode/time interval")
                state_raw = np.asarray(table["observation.state"][offset].as_py(), dtype=np.float32)
                action_raw = np.asarray(table["action"].slice(offset, 16).to_pylist(), dtype=np.float32)
                normalized = normalize_and_pad_trex_state_action(state_raw, action_raw, stats)
                state_out[row] = normalized["state"]
                action_out[row] = normalized["action"]
        state_out.flush(); action_out.flush()
        result[public] = {"count": len(episode), "data_files": len(grouped)}
    return result


def action_phase(args: argparse.Namespace, spec: Mapping[str, Any], cache_root: Path, device: torch.device) -> dict[str, Any]:
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required for Action")
    source = ReleasedTokenizerSource.open(args.unit_checkpoint / "tokenizer")
    if source.old_rows_digest() != spec["frozen_identity"]["old_action_rows_digest"]:
        raise RuntimeError("Original UniT Action rows changed")
    model, metadata = load_shared_transition_checkpoint(args.action_checkpoint, source)
    model.eval().requires_grad_(False).to(device)
    payload = torch.load(args.action_checkpoint, map_location="cpu", weights_only=False)
    if payload["feature_stats"].get("canonical_sha256") != spec["frozen_identity"]["action_feature_stats_canonical_sha256"]:
        raise RuntimeError("A-R transition feature statistics changed")
    state_summary = extract_state_action(args, cache_root)
    result: dict[str, Any] = {"state_action": state_summary, "checkpoint_metadata": metadata, "splits": {}}
    with torch.inference_mode():
        for public in PUBLIC_TO_SOURCE:
            if args.only_split is not None and public != args.only_split:
                continue
            split_root = cache_root / public
            state = np.load(split_root / "state.npy", mmap_mode="r")
            action = np.load(split_root / "action.npy", mmap_mode="r")
            z_a = np.lib.format.open_memmap(split_root / "z_a.npy", mode="r+")
            for start in range(0, len(state), args.batch_size):
                stop = min(start + args.batch_size, len(state))
                state_batch = torch.from_numpy(np.array(state[start:stop], copy=True)).to(device)
                action_batch = torch.from_numpy(np.array(action[start:stop], copy=True)).to(device)
                embodiment = torch.full((stop - start,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
                z_a[start:stop] = model.encode(state_batch, action_batch, embodiment)[0].float().cpu().numpy()
            canonical_reused = 0
            if public == "test":
                pair_ids = np.load(split_root / "pair_id.npy", mmap_mode="r", allow_pickle=False)
                row_by_id = {str(value): row for row, value in enumerate(pair_ids)}
                with np.load(args.integration_cache, allow_pickle=False) as accepted:
                    accepted_ids = [str(value) for value in accepted["pair_id"]]
                    if len(accepted_ids) != 960 or len(set(accepted_ids)) != 960:
                        raise RuntimeError("accepted integration cache must contain 960 unique rows")
                    try:
                        rows = np.asarray([row_by_id[value] for value in accepted_ids], dtype=np.int64)
                    except KeyError as error:
                        raise RuntimeError("C1 test cache is missing an accepted canonical pair") from error
                    if not np.array_equal(np.asarray(state[rows]), accepted["state"]):
                        raise RuntimeError("accepted canonical Action state identity changed")
                    if not np.array_equal(np.asarray(action[rows]), accepted["action"]):
                        raise RuntimeError("accepted canonical Action window identity changed")
                    if accepted["z_a"].shape != (960, 8, 32) or not np.isfinite(accepted["z_a"]).all():
                        raise RuntimeError("accepted canonical Action latents are invalid")
                    z_a[rows] = accepted["z_a"]
                    canonical_reused = len(rows)
            z_a.flush()
            check = min(args.batch_size, len(state))
            state_batch = torch.from_numpy(np.array(state[:check], copy=True)).to(device)
            action_batch = torch.from_numpy(np.array(action[:check], copy=True)).to(device)
            embodiment = torch.full((check,), TREX_EMBODIMENT_ID, dtype=torch.long, device=device)
            first = model.encode(state_batch, action_batch, embodiment)[0]
            second = model.encode(state_batch, action_batch, embodiment)[0]
            if not torch.equal(first, second) or not np.isfinite(z_a).all():
                raise RuntimeError("A-R cache extraction is non-deterministic or non-finite")
            result["splits"][public] = {
                "count": len(state),
                "repeat_exact": True,
                "accepted_canonical_rows_reused": canonical_reused,
            }
    phase_marker(cache_root, "action", result)
    return result


def load_reusable_vision() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for worktree in worktree_paths():
        for public in PUBLIC_TO_SOURCE:
            path = worktree / ".local/cache/tactile_unit/c0" / f"paired_{public}.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as payload:
                for pair, latent in zip(payload["pair_id"], payload["z_v"]):
                    key = str(pair)
                    value = np.asarray(latent, dtype=np.float32)
                    if key in result and not np.array_equal(result[key], value):
                        raise RuntimeError("conflicting identity-matched C0 Vision cache rows")
                    result[key] = value
    return result


def decode_cluster(path: Path, targets: Iterable[float]) -> dict[float, np.ndarray]:
    import av

    requested = sorted(set(map(float, targets)))
    chosen: dict[float, tuple[float, np.ndarray]] = {}
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = max(0.0, requested[0] - 2.0 / CANONICAL_FPS)
        if stream.time_base is not None:
            container.seek(int(start / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
        last = requested[-1]
        for frame in container.decode(stream):
            timestamp = float(frame.time) if frame.time is not None else None
            if timestamp is None:
                continue
            insertion = int(np.searchsorted(requested, timestamp))
            for position in (insertion - 1, insertion):
                if 0 <= position < len(requested):
                    target = requested[position]
                    distance = abs(timestamp - target)
                    if target not in chosen or distance < chosen[target][0]:
                        chosen[target] = (distance, frame.to_ndarray(format="rgb24"))
            if timestamp > last + 2.0 / CANONICAL_FPS:
                break
    if len(chosen) != len(requested):
        raise RuntimeError(f"sparse decoder missed requested frames in {path.name}")
    if any(distance > 1.0 / CANONICAL_FPS + 1e-4 for distance, _ in chosen.values()):
        raise RuntimeError("sparse decoder returned a non-nearest frame")
    return {target: value[1] for target, value in chosen.items()}


def clustered_rows(
    current_time: list[float],
    future_time: list[float],
    gap: float,
    *,
    maximum_pairs: int = 32,
    maximum_span: float = 4.0,
) -> list[list[int]]:
    order = sorted(range(len(current_time)), key=lambda index: current_time[index])
    groups: list[list[int]] = []
    for index in order:
        start_new = not groups
        if groups:
            start_new = (
                len(groups[-1]) >= maximum_pairs
                or current_time[index] - max(future_time[row] for row in groups[-1]) > gap
                or future_time[index] - current_time[groups[-1][0]] > maximum_span
            )
        if start_new:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def vision_phase(args: argparse.Namespace, spec: Mapping[str, Any], cache_root: Path, device: torch.device) -> dict[str, Any]:
    if args.unit_checkpoint is None:
        raise RuntimeError("--unit-checkpoint or UNIT_FULLDATA_CKPT is required for Vision")
    vision_spec = {"frozen_identity": {"original_unit_tokenizer_files_sha256": spec["frozen_identity"]["original_unit_tokenizer_files_sha256"]}}
    vision, identity = load_frozen_vision(args.unit_checkpoint, vision_spec, device)
    pointers = {item.episode_id: item for item in load_episode_video_pointers(args.dataset_root)}
    reusable = load_reusable_vision()
    result: dict[str, Any] = {"identity": identity, "splits": {}}
    with torch.inference_mode():
        for public in PUBLIC_TO_SOURCE:
            split_root = cache_root / public
            pair_ids = np.load(split_root / "pair_id.npy", mmap_mode="r", allow_pickle=False)
            episode = np.load(split_root / "episode_id.npy", mmap_mode="r")
            anchor = np.load(split_root / "t.npy", mmap_mode="r")
            z_v = np.lib.format.open_memmap(split_root / "z_v.npy", mode="r+")
            complete = np.zeros(len(pair_ids), dtype=bool)
            for row, key in enumerate(map(str, pair_ids)):
                if key in reusable:
                    z_v[row] = reusable[key]
                    complete[row] = True
            grouped: dict[str, list[int]] = defaultdict(list)
            for row in np.flatnonzero(~complete):
                grouped[pointers[int(episode[row])].relative_path].append(int(row))
            decoded_pairs = 0
            processed_clusters = 0
            for relative_path, rows in sorted(grouped.items()):
                current_time = [pointers[int(episode[row])].from_timestamp + int(anchor[row]) / CANONICAL_FPS for row in rows]
                future_time = [value + 16.0 / CANONICAL_FPS for value in current_time]
                for group in clustered_rows(current_time, future_time, args.vision_cluster_gap):
                    target_times = [current_time[index] for index in group] + [future_time[index] for index in group]
                    frames = decode_cluster(args.dataset_root / relative_path, target_times)
                    for batch_start in range(0, len(group), args.batch_size):
                        batch_group = group[batch_start:batch_start + args.batch_size]
                        obs = np.stack([preprocess_trex_rgb(frames[current_time[index]]) for index in batch_group])
                        goal = np.stack([preprocess_trex_rgb(frames[future_time[index]]) for index in batch_group])
                        obs_tensor = torch.from_numpy(obs)[:, None].to(device, dtype=vision.dtype)
                        goal_tensor = torch.from_numpy(goal)[:, None].to(device, dtype=vision.dtype)
                        values, _, _ = vision.vision_branch(obs_tensor, goal_tensor, batch_size=len(batch_group))
                        latent = vision.vq_down_resampler(values).float().cpu().numpy()
                        destination = [rows[index] for index in batch_group]
                        z_v[destination] = latent
                        complete[destination] = True
                        decoded_pairs += len(destination)
                    del frames, obs, goal, obs_tensor, goal_tensor, values, latent
                    processed_clusters += 1
                    if processed_clusters % 32 == 0:
                        gc.collect()
                        try:
                            ctypes.CDLL(None).malloc_trim(0)
                        except (AttributeError, OSError):
                            pass
            z_v.flush()
            if not complete.all() or not np.isfinite(z_v).all():
                raise RuntimeError(f"Vision cache incomplete for {public}")
            result["splits"][public] = {
                "count": len(pair_ids),
                "reused": int(len(pair_ids) - decoded_pairs),
                "decoded": decoded_pairs,
                "video_files": len(grouped),
            }
    phase_marker(cache_root, "vision", result)
    return result


def finalize(args: argparse.Namespace, spec: Mapping[str, Any], cache_root: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    for phase in ("contact", "action", "vision"):
        if not (cache_root / "phases" / f"{phase}.json").is_file():
            raise RuntimeError(f"cannot finalize before {phase} extraction")
    splits = {
        public: split_manifest(cache_root / public, cache_root, int(spec["cached_pairs"][public]))
        for public in PUBLIC_TO_SOURCE
    }
    manifest: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "implementation_version": spec["runtime"]["implementation_version"],
        "format": spec["runtime"]["format"],
        "dtype": "float32",
        "transition_shape": [8, 32],
        "horizon_frames": 16,
        "sample_order": "ascending frozen S2 source_index within split",
        "selection_manifest_sha256": sha256_file(cache_root / "selection_manifest.json"),
        "provenance": dict(provenance),
        "splits": splits,
    }
    manifest["canonical_sha256"] = sha256_json(manifest)
    atomic_json(cache_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    spec = json.loads(args.config.read_text())
    cache_root = args.cache_root or ROOT / spec["runtime"]["cache_root"]
    cache_root.mkdir(parents=True, exist_ok=True)
    set_seed(int(spec["seed"]))
    provenance = verify_sources(args, spec)
    selection = freeze_selection(args, spec, cache_root)
    provenance.update({
        "selection_manifest_canonical_sha256": selection["canonical_sha256"],
        "vision_preprocessing_identity": spec["frozen_identity"]["vision_preprocessing_identity"],
        "action_feature_stats_canonical_sha256": spec["frozen_identity"]["action_feature_stats_canonical_sha256"],
        "s1_teacher_checkpoint_sha256": spec["frozen_identity"]["s1_teacher_checkpoint_sha256"],
        "s2_encoder_parameter_digest": spec["frozen_identity"]["s2_encoder_parameter_digest"],
        "s2_decoder_parameter_digest": spec["frozen_identity"]["s2_decoder_parameter_digest"],
        "old_action_rows_digest": spec["frozen_identity"]["old_action_rows_digest"],
        "dtype": "float32",
        "shape": [8, 32],
        "k": 16,
    })
    if args.phase == "selection":
        print(json.dumps(selection, indent=2, sort_keys=True))
        return
    device, lock_handle, gpu = resolve_device(args.device)
    try:
        results: dict[str, Any] = {"gpu": gpu}
        if args.phase in {"contact", "all"}:
            results["contact"] = contact_phase(args, spec, cache_root, device)
        if args.phase in {"action", "all"}:
            results["action"] = action_phase(args, spec, cache_root, device)
        if args.phase in {"vision", "all"}:
            results["vision"] = vision_phase(args, spec, cache_root, device)
        if args.phase in {"finalize", "all"}:
            results["manifest"] = finalize(args, spec, cache_root, provenance)
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
