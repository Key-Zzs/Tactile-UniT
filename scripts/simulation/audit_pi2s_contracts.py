#!/usr/bin/env python3
"""Freeze and validate the bounded, read-only S4.3-PI2S protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "configs/simulation/pi2s/diagnostic_protocol.json"
BASE_HEAD = "6602c05d78ff97a150255cc37aae83e1f15c9df6"
CONTACT_SHA = "833db9ddb4d37534bf38a7ed0b214fee2f000bb4e489d3fa9507b4e7bca5bd8e"
VA27_SHA = "7d51a23672273ec3ea46f947080c0c3ea66db55332522b199caab00ad4fc2126"
DEV_MANIFEST_SHA = "d47853046c8293fe062eb8946b4b7c51a0f5953f6bd22bbc05b8029914ef96bd"
OFFICIAL_DATASET_MANIFEST_SHA = "022057f7575211b8c099da7a04bcc03bdb11bd2830c1878be2aad88345bd5f1b"
RAW_DATASET_MANIFEST_SHA = "8aa60768a0d8f2c64332215c75acef7ec35fd456262f2e418b83b56aced046b2"
ALIGNMENT_MANIFEST_SHA = "ee817ec97d4a5b9ca485548107bf78cf1163dd351e154a9390a30d3e730f1b7f"
POLICY_DEV_MANIFEST_SHA = "576080b8c19265e965ab593f74be053d5646917d1752a967c438e76f83157efa"
POLICY_DEV_CACHE_MANIFEST_SHA = "aa3a2e039c146849b7cb9824f75cd619ea8bcb5c184c26a799f46c642cd5a1f3"
INTERMEDIATE_HASH_MANIFEST_SHA = "3fc3d42651a7128c54aa403f927a36561ff2e70b831a4e52e0636f02a79a867f"
ET_CHECKPOINT_SHA = "3d4519195e7a0d9f5af63399840a48121d5f614ea587c80b9461fc696a372a19"
OPENPI_CLIENT_INIT_SHA = "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed"
OPENPI_CLIENT_IMAGE_TOOLS_SHA = "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
OPENPI_CLIENT_PYTHON_FILES = 13
OPENPI_CLIENT_PYTHON_TREE_SHA = "92fa31ab5335cd7613d656fcea70569e7797a73f6584f896ab7ff45a3ebf5cb4"
PALIGEMMA_TOKENIZER_SHA = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
PALIGEMMA_TOKENIZER_BYTES = 4_264_023
PALIGEMMA_TOKENIZER_SNAPSHOT = "source_snapshots/data/paligemma_tokenizer.model"
PALIGEMMA_TOKENIZER_MANIFEST = "source_snapshots/data/paligemma_tokenizer.manifest.json"
PALIGEMMA_TOKENIZER_MANIFEST_SHA = (
    "7712fbc4fcc3948b50de0176d3403ea07ac75518462074e43bf96ffbff57cd92"
)
OPENPI_ENVIRONMENT_FREEZE_SHA = "39664447f71898e7ae8c0d7e41a79d74b13fe009546a8efcd69504ca64c91e89"
SOURCE_IDENTITIES = {
    "gr00t/simulation/s4_3_pi1.py": "11ace9ee717e859d6dd3a18691a9ef087e97cb4a47ad2d42fc43f4ef381017dc",
    "gr00t/simulation/s4_3_act.py": "b8d45a0c63862005431154cabdce5c771fe2259286371d070a9a7eaaf8ad0bc4",
    "gr00t/simulation/pi1d_runtime.py": "fa8ac52e84f7d203ad4dde09184b2ec6c0664b1a1c4b2610b24991cc9b7b09ad",
    "scripts/simulation/build_s4_3_pi1_sidecar.py": (
        "d53fd758023d315380f7d5a85e373cbb49c138d4fe9994a8d2ec38579086643b"
    ),
    "scripts/simulation/serve_s4_3_pi1_contact_state.py": (
        "87ca113f340f195f4bb761bc47b944db5869379383aa34e5f9fd4cb8777f27cf"
    ),
    "gr00t/simulation/pi05_tactile_unit.py": "d5ab1affbfbc16f708e9bcdf74d5dadac8c3730abb61f590de356e1292a90632",
    "gr00t/simulation/sim_contact_models.py": (
        "5c397c24127c98ad3e6ccd2223ed5fd01c5974edbaedc9c815123965e31548f4"
    ),
    "gr00t/tactile_teacher/models.py": (
        "9ba4523c7c0aa7438da9a709ea0e1884be0dd56c7526641bd943d64a098afda6"
    ),
}
DIAGNOSTIC_IMPLEMENTATION_PATHS = {
    "analyze_pi2s_historical_evidence": "scripts/simulation/analyze_pi2s_historical_evidence.py",
    "audit_pi2s_initialization_rng": "scripts/simulation/audit_pi2s_initialization_rng.py",
    "hash_pi2s_intermediate_checkpoints": (
        "scripts/simulation/hash_pi2s_intermediate_checkpoints.py"
    ),
    "run_pi2s_h_parity": "scripts/simulation/run_pi2s_h_parity.py",
    "run_pi2s_model_diagnostics": "scripts/simulation/run_pi2s_model_diagnostics.py",
}
TASK_PROMPT = "Grasp the tongs and perform three consecutive open-close motions."
INPUT_SNAPSHOT_VERSION = "pi2s_protocol_v3"
FOCUS = (
    ("B0", 43),
    ("B_VA27", 43),
    ("B1", 43),
    ("B_HVA", 42),
    ("B_HVA", 43),
    ("B2", 43),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def python_tree_sha(root: Path) -> tuple[int, str]:
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_json_fsync(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_npy_fsync(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def copy_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, 8 * 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: {actual} != {expected}")


def experiment_root() -> Path:
    path = (ROOT / ".local/experiments").resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError("experiment root is not a directory")
    return path


def symbolic(path: Path, experiments: Path) -> str:
    path = path.resolve()
    try:
        relative = path.relative_to(experiments)
    except ValueError as exc:
        raise RuntimeError(f"path is outside experiment root: {path}") from exc
    return "$EXPERIMENT_ROOT/" + relative.as_posix()


def symbolic_repo(path: Path) -> str:
    path = path.resolve()
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository root: {path}") from exc
    return "$REPO_ROOT/" + relative.as_posix()


def stable_identity_key(identity: dict[str, Any]) -> str:
    """Hash only frozen identity fields, never values or outcome labels."""
    return canonical_sha(identity)


def select_train_rows(
    episode: np.ndarray,
    frame: np.ndarray,
    control_tick: np.ndarray,
    valid: np.ndarray,
    episode_lengths: list[int],
    raw_trim: list[int],
    source_groups: list[str],
) -> tuple[list[int], dict[int, str], dict[int, dict[str, Any]]]:
    full_action_horizon = np.asarray(
        [
            int(frame[index]) + 29 < episode_lengths[int(episode[index])]
            for index in range(len(frame))
        ],
        dtype=bool,
    )
    candidates = np.flatnonzero(valid & full_action_horizon).tolist()
    identities: dict[int, dict[str, Any]] = {}
    keys: dict[int, str] = {}
    for index in candidates:
        episode_index = int(episode[index])
        frame_index = int(frame[index])
        tick = int(control_tick[index])
        expected_tick = int(raw_trim[episode_index]) + frame_index
        if tick != expected_tick:
            raise RuntimeError(f"TRAIN raw tick mismatch at row {index}: {tick} != {expected_tick}")
        identity = {
            "split": "TRAIN",
            "official_dataset_manifest_sha256": OFFICIAL_DATASET_MANIFEST_SHA,
            "contact_sidecar_sha256": CONTACT_SHA,
            "alignment_manifest_sha256": ALIGNMENT_MANIFEST_SHA,
            "source_group_id": source_groups[episode_index],
            "episode_index": episode_index,
            "frame_index": frame_index,
            "row": index,
            "control_tick_end": tick,
            "history_control_ticks_inclusive": [max(int(raw_trim[episode_index]), tick - 25), tick],
            "history_left_repeat_count": max(0, 25 - frame_index),
            "action_control_ticks_inclusive": [tick, tick + 29],
            "auxiliary_target_control_tick": tick + 27,
        }
        identities[index] = identity
        keys[index] = stable_identity_key(identity)
    by_group: dict[str, list[int]] = {}
    for index in candidates:
        group = identities[index]["source_group_id"]
        by_group.setdefault(group, []).append(index)
    if len(by_group) != 100:
        raise RuntimeError(f"expected 100 true TRAIN source groups, found {len(by_group)}")
    selected = [min(rows, key=keys.__getitem__) for _, rows in sorted(by_group.items())]
    selected_set = set(selected)
    remaining = sorted((row for row in candidates if row not in selected_set), key=keys.__getitem__)
    selected.extend(remaining[: 512 - len(selected)])
    selected.sort(key=keys.__getitem__)
    if len(selected) != 512 or len(set(selected)) != 512:
        raise RuntimeError("TRAIN selection cardinality failure")
    return selected, keys, identities


def select_policy_dev_rows(
    cache_root: Path, membership: dict[str, Any]
) -> tuple[list[int], dict[int, str], dict[int, dict[str, Any]], list[str]]:
    episode_file = cache_root / "episodes.json"
    episodes = json.loads(episode_file.read_text())
    episode_ids = episodes["episode_ids"]
    if episodes.get("task") != "pinch_tongs" or episodes.get("split") != "dev":
        raise RuntimeError("unexpected policy-development episode membership")
    rows_by_attempt = {
        row["attempt_id"]: row
        for row in membership["successful_dev"]
        if row["task"] == "pinch_tongs"
    }
    if len(episode_ids) != 25 or set(episode_ids) != set(rows_by_attempt):
        raise RuntimeError("policy-development cache/membership attempt mismatch")
    episode_index = np.load(cache_root / "episode_index.npy", mmap_mode="r", allow_pickle=False)
    tick = np.load(cache_root / "t.npy", mmap_mode="r", allow_pickle=False)
    pair_id = np.load(cache_root / "pair_id.npy", mmap_mode="r", allow_pickle=False)
    if episode_index.shape != (8815,) or tick.shape != (8815,) or pair_id.shape != (8815,):
        raise RuntimeError("unexpected policy-development cache shapes")
    identities: dict[int, dict[str, Any]] = {}
    keys: dict[int, str] = {}
    for index in range(8815):
        attempt_id = episode_ids[int(episode_index[index])]
        source = rows_by_attempt[attempt_id]
        current_tick = int(tick[index])
        identity = {
            "split": "POLICY_DEV",
            "membership_manifest_sha256": POLICY_DEV_MANIFEST_SHA,
            "cache_manifest_sha256": POLICY_DEV_CACHE_MANIFEST_SHA,
            "source_dataset_revision": membership["source_dataset_revision"],
            "source_group_id": source["source_group_id"],
            "source_group_index": int(source["source_group_index"]),
            "attempt_id": attempt_id,
            "attempt_steps_npz_sha256": source["steps_npz_sha256"],
            "cache_row": index,
            "pair_id": pair_id[index].decode("ascii").rstrip("\x00"),
            "current_control_tick": current_tick,
            "history_control_ticks_inclusive": [current_tick - 25, current_tick],
            "action_control_ticks_inclusive": [current_tick, current_tick + 26],
            "contact_target_control_tick": current_tick + 27,
        }
        selection_identity = {
            key: value for key, value in identity.items() if key != "attempt_steps_npz_sha256"
        }
        identity["selection_identity_sha256"] = canonical_sha(selection_identity)
        identities[index] = identity
        keys[index] = stable_identity_key(selection_identity)
    by_group: dict[str, list[int]] = {}
    for index, identity in identities.items():
        by_group.setdefault(identity["source_group_id"], []).append(index)
    if len(by_group) != 5:
        raise RuntimeError(f"expected five policy-development source groups, found {len(by_group)}")
    selected = [min(rows, key=keys.__getitem__) for _, rows in sorted(by_group.items())]
    selected_set = set(selected)
    selected.extend(
        sorted((row for row in identities if row not in selected_set), key=keys.__getitem__)[
            : 256 - len(selected)
        ]
    )
    selected.sort(key=keys.__getitem__)
    if len(selected) != 256 or len(set(selected)) != 256:
        raise RuntimeError("POLICY_DEV selection cardinality failure")
    return selected, keys, identities, episode_ids


def select_balanced(
    rows: list[int], keys: dict[int, str], tactile: np.ndarray, per_class: int
) -> list[int]:
    active = sorted(
        (row for row in rows if np.linalg.norm(tactile[row].astype(np.float64)) > 0.0),
        key=keys.__getitem__,
    )
    inactive = sorted(
        (row for row in rows if np.linalg.norm(tactile[row].astype(np.float64)) == 0.0),
        key=keys.__getitem__,
    )
    if len(active) < per_class or len(inactive) < per_class:
        raise RuntimeError("insufficient fixed contact strata")
    return sorted(active[:per_class] + inactive[:per_class], key=keys.__getitem__)


def snapshot_file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"snapshot must not contain symlinks: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def verify_input_snapshot(
    root: Path, train_selection_sha: str, policy_dev_selection_sha: str
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing frozen input snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tactile3d-unit.s4-3-pi2s-input-snapshot.v1":
        raise RuntimeError("unexpected PI2S input snapshot schema")
    if manifest.get("train_selection_sha256") != train_selection_sha:
        raise RuntimeError("TRAIN selection differs from frozen input snapshot")
    if manifest.get("policy_dev_selection_sha256") != policy_dev_selection_sha:
        raise RuntimeError("POLICY_DEV selection differs from frozen input snapshot")
    actual = snapshot_file_records(root)
    if actual != manifest.get("files"):
        raise RuntimeError("PI2S input snapshot file manifest mismatch")
    return manifest


def prepare_input_snapshot(
    *,
    pi2s_root: Path,
    contact_path: Path,
    va27_path: Path,
    train_selected: list[int],
    train_identities: dict[int, dict[str, Any]],
    fixed_rows: list[int],
    policy_cache_root: Path,
    policy_selected: list[int],
    policy_identities: dict[int, dict[str, Any]],
    policy_episode_ids: list[str],
    policy_membership: dict[str, Any],
) -> dict[str, Any]:
    train_selection_sha = canonical_sha([train_identities[row] for row in train_selected])
    policy_selection_sha = canonical_sha([policy_identities[row] for row in policy_selected])
    final_root = pi2s_root / f"source_snapshots/{INPUT_SNAPSHOT_VERSION}"
    if final_root.exists():
        return verify_input_snapshot(final_root, train_selection_sha, policy_selection_sha)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{INPUT_SNAPSHOT_VERSION}.staging.", dir=final_root.parent)
    )
    train_root = staging / "train_fixed"
    policy_root = staging / "policy_dev"
    source_root = staging / "source_manifests"
    train_root.mkdir()
    policy_root.mkdir()
    source_root.mkdir()

    source_manifests = {
        "official_dataset_manifest.json": (
            ROOT / ".local/artifacts/simulation/s4_3_pi0/official_dataset_manifest.json"
        ),
        "raw_dataset_manifest.json": ROOT
        / ".local/artifacts/simulation/s4_3_pi1/raw_dataset_manifest.json",
        "official_dataset_alignment.json": (
            ROOT / ".local/artifacts/simulation/s4_3_pi1/official_dataset_alignment.json"
        ),
        "policy_expert_dataset_manifest.json": (
            ROOT / ".local/artifacts/simulation/s4_3_pd/policy_expert_dataset_manifest.json"
        ),
        "policy_expert_cache_manifest.json": ROOT
        / ".local/cache/simulation/s4_3_restart/manifest.json",
    }
    for name, source in source_manifests.items():
        copy_fsync(source, source_root / name)
    fsync_directory(source_root)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError as exc:
        raise RuntimeError(
            "--prepare-inputs requires the audited OpenPI interpreter with lerobot.datasets"
        ) from exc
    dataset_root = (
        ROOT / ".local/external/s4_3_pi0/datasets/DexJoCo-Datasets-LeRobot/"
        "dexjoco_lerobot_datasets/pinch_tongs"
    ).resolve(strict=True)
    metadata = LeRobotDatasetMetadata("local_repo", root=dataset_root)
    if metadata.fps != 30 or metadata.total_frames != 40065:
        raise RuntimeError("official LeRobot metadata identity mismatch")
    dataset = LeRobotDataset(
        "local_repo",
        root=dataset_root,
        delta_timestamps={"action": [index / metadata.fps for index in range(30)]},
    )
    fixed_count = len(fixed_rows)
    front = np.empty((fixed_count, 3, 640, 640), dtype=np.uint8)
    wrist = np.empty_like(front)
    state = np.empty((fixed_count, 23), dtype=np.float32)
    action = np.empty((fixed_count, 30, 22), dtype=np.float32)
    action_is_pad = np.empty((fixed_count, 30), dtype=bool)
    timestamps = np.empty(fixed_count, dtype=np.float32)
    with (
        np.load(contact_path, allow_pickle=False) as contact,
        np.load(va27_path, allow_pickle=False) as va27,
    ):
        np.testing.assert_array_equal(contact["index"], va27["index"])
        fixed_contact_state = contact["contact_state"][fixed_rows].astype(np.float32, copy=True)
        fixed_tactile_history = contact["tactile_history"][fixed_rows].astype(np.float32, copy=True)
        fixed_contact_target = contact["contact_shared_target"][fixed_rows].astype(
            np.float32, copy=True
        )
        fixed_contact_valid = contact["physical_aux_valid"][fixed_rows].astype(bool, copy=True)
        fixed_va_target = va27["va_shared_target"][fixed_rows].astype(np.float32, copy=True)
        fixed_va_valid = va27["va_aux_valid"][fixed_rows].astype(bool, copy=True)
        np.testing.assert_array_equal(fixed_contact_valid, fixed_va_valid)
        fixed_episode = contact["episode_index"][fixed_rows].astype(np.int64, copy=True)
        fixed_frame = contact["frame_index"][fixed_rows].astype(np.int64, copy=True)
        fixed_control_tick = contact["control_tick_end"][fixed_rows].astype(np.int64, copy=True)
    for output_index, row_index in enumerate(fixed_rows):
        sample = dataset[row_index]
        if sample["task"] != TASK_PROMPT:
            raise RuntimeError(f"unexpected prompt for official row {row_index}")
        for key, destination in (
            ("observation.images.front", front),
            ("observation.images.wrist", wrist),
        ):
            image = sample[key].detach().cpu().numpy()
            encoded = np.rint(image * 255.0).astype(np.uint8)
            if not np.array_equal(image, encoded.astype(np.float32) / 255.0):
                raise RuntimeError(f"non-lossless uint8 image conversion at row {row_index}: {key}")
            destination[output_index] = encoded
        state[output_index] = sample["observation.state"].detach().cpu().numpy()
        action[output_index] = sample["action"].detach().cpu().numpy()
        action_is_pad[output_index] = sample["action_is_pad"].detach().cpu().numpy()
        timestamps[output_index] = float(sample["timestamp"].item())
        if bool(action_is_pad[output_index].any()):
            raise RuntimeError(f"fixed row lacks full official 30-action horizon: {row_index}")
        if int(sample["index"].item()) != row_index:
            raise RuntimeError(f"official dataset row identity mismatch: {row_index}")
        if int(sample["episode_index"].item()) != int(fixed_episode[output_index]):
            raise RuntimeError(f"official/sidecar episode mismatch: {row_index}")
        if int(sample["frame_index"].item()) != int(fixed_frame[output_index]):
            raise RuntimeError(f"official/sidecar frame mismatch: {row_index}")
    train_arrays = {
        "rows.npy": np.asarray(fixed_rows, dtype=np.int64),
        "front_rgb_chw_uint8.npy": front,
        "wrist_rgb_chw_uint8.npy": wrist,
        "state.npy": state,
        "action.npy": action,
        "action_is_pad.npy": action_is_pad,
        "timestamp_converter_label_sec.npy": timestamps,
        "episode_index.npy": fixed_episode,
        "frame_index.npy": fixed_frame,
        "control_tick_end_raw50hz.npy": fixed_control_tick,
        "contact_state.npy": fixed_contact_state,
        "tactile_history.npy": fixed_tactile_history,
        "contact_target_t_plus_27.npy": fixed_contact_target,
        "contact_aux_valid.npy": fixed_contact_valid,
        "va_target_t_plus_27.npy": fixed_va_target,
        "va_aux_valid.npy": fixed_va_valid,
    }
    for name, value in train_arrays.items():
        write_npy_fsync(train_root / name, value)
    write_json_fsync(
        train_root / "identity.json",
        {
            "task_prompt": TASK_PROMPT,
            "converter_fps_label": 30,
            "raw_control_hz": 50,
            "rows": [train_identities[row] for row in fixed_rows],
            "limitations": (
                "Decoded converter frames are losslessly stored as uint8. The 30 Hz dataset label is "
                "not evidence of raw 50-to-30 Hz physical resampling."
            ),
        },
    )
    fsync_directory(train_root)

    policy_array_names = (
        "episode_index",
        "t",
        "pair_id",
        "proprio",
        "action",
        "tactile_history",
        "contact_state",
        "contact_target",
        "vision",
        "contact_complete",
        "vision_complete",
    )
    selected_array = np.asarray(policy_selected, dtype=np.int64)
    write_npy_fsync(policy_root / "cache_rows.npy", selected_array)
    for name in policy_array_names:
        source = np.load(policy_cache_root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        write_npy_fsync(policy_root / f"{name}.npy", np.asarray(source[selected_array]))

    attempt_rows = {
        row["attempt_id"]: row
        for row in policy_membership["successful_dev"]
        if row["task"] == "pinch_tongs"
    }
    source_attempt_root = ROOT / ".local/datasets/simulation/s4_3_policy_expert/attempts"
    selected_by_attempt: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for selection_position, row in enumerate(policy_selected):
        identity = policy_identities[row]
        selected_by_attempt.setdefault(identity["attempt_id"], []).append(
            (selection_position, identity)
        )
    image_by_position: dict[int, tuple[bytes, str, int, int]] = {}
    metadata_copies = []
    for attempt_id, selected_rows in sorted(selected_by_attempt.items()):
        source_attempt = source_attempt_root / attempt_id
        source_metadata_path = source_attempt / "metadata.json"
        source_metadata = json.loads(source_metadata_path.read_text())
        expected = attempt_rows[attempt_id]
        verify_sha(
            source_attempt / "steps.npz", expected["steps_npz_sha256"], f"{attempt_id} steps"
        )
        with np.load(source_attempt / "steps.npz", allow_pickle=False) as steps:
            for position, identity in selected_rows:
                tick = identity["current_control_tick"]
                if int(steps["control_step"][tick]) != tick:
                    raise RuntimeError(f"non-identity control step in {attempt_id} at {tick}")
                relative = str(steps["rgb_reference"][tick])
                image_path = source_attempt / relative
                image_bytes = image_path.read_bytes()
                image_by_position[position] = (
                    image_bytes,
                    relative,
                    tick,
                    int(source_metadata["metadata"]["source_group_index"]),
                )
        metadata_copies.append(
            {
                "membership": expected,
                "metadata_sha256": sha256_file(source_metadata_path),
                "metadata": source_metadata,
            }
        )
    image_records = []
    offset = 0
    with (policy_root / "current_random_camera_jpeg_bytes.bin").open("xb") as stream:
        for position in range(len(policy_selected)):
            image_bytes, relative, tick, group_index = image_by_position[position]
            stream.write(image_bytes)
            image_records.append(
                {
                    "selection_position": position,
                    "cache_row": policy_selected[position],
                    "attempt_id": policy_identities[policy_selected[position]]["attempt_id"],
                    "source_group_index": group_index,
                    "control_tick": tick,
                    "source_relative_path": relative,
                    "offset": offset,
                    "bytes": len(image_bytes),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            )
            offset += len(image_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    write_json_fsync(policy_root / "current_random_camera_jpeg_index.json", image_records)
    write_json_fsync(policy_root / "selected_attempt_metadata.json", metadata_copies)
    write_json_fsync(
        policy_root / "identity.json",
        {
            "rows": [policy_identities[row] for row in policy_selected],
            "available_observation_fields": [
                "one random_camera current JPEG",
                "proprio[22]",
                "tactile_history[26,30]",
                "contact_state[256]",
                "action[27,22]",
                "contact_target[8,32] at t+27",
            ],
            "missing_for_official_pi05_fixed_forward": [
                "front+wrist paired cameras",
                "state[23]",
                "task prompt field",
                "action horizon[30,22]",
            ],
            "scientific_scope": "H/action/domain distribution only; not an official pi0.5 fixed observation",
        },
    )
    fsync_directory(policy_root)
    fsync_directory(staging)

    manifest = {
        "schema": "tactile3d-unit.s4-3-pi2s-input-snapshot.v1",
        "status": "COMPLETE_VERIFIED",
        "train_selection_sha256": train_selection_sha,
        "policy_dev_selection_sha256": policy_selection_sha,
        "train_fixed_rows": len(fixed_rows),
        "policy_dev_rows": len(policy_selected),
        "policy_dev_source_groups": len(
            {policy_identities[row]["source_group_id"] for row in policy_selected}
        ),
        "policy_dev_attempts": len(
            {policy_identities[row]["attempt_id"] for row in policy_selected}
        ),
        "files": snapshot_file_records(staging),
    }
    write_json_fsync(staging / "manifest.json", manifest)
    fsync_directory(staging)
    os.rename(staging, final_root)
    fsync_directory(final_root.parent)
    return verify_input_snapshot(final_root, train_selection_sha, policy_selection_sha)


def build_protocol(*, prepare_inputs: bool = False) -> dict[str, Any]:
    experiments = experiment_root()
    track_a = experiments / "simulation/s4_3_pi2b_policy"
    freeze_path = track_a / "artifacts/pre_final_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    all_checkpoints = []
    checkpoint_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in freeze["checkpoints"]:
        if Path(row["path"]).name != "29999":
            raise RuntimeError(f"unexpected final checkpoint directory: {row['path']}")
        checkpoint = {
            "model": row["model"],
            "training_seed": int(row["training_seed"]),
            "path": symbolic(Path(row["path"]), experiments),
            "tree_sha256": row["tree_sha256"],
            "hash_source": row["hash_source"],
            "params_present": bool(row["params_present"]),
            "train_state_present": bool(row["train_state_present"]),
            "directory_step": 29999,
            "expected_restored_train_step": 30000,
        }
        all_checkpoints.append(checkpoint)
        checkpoint_by_key[(checkpoint["model"], checkpoint["training_seed"])] = checkpoint
    if len(all_checkpoints) != 15 or len(checkpoint_by_key) != 15:
        raise RuntimeError("expected exactly fifteen final checkpoints")
    focus = [checkpoint_by_key[key] for key in FOCUS]

    pi2s_root = experiments / "simulation/s4_3_pi2s"
    intermediate_manifest_path = pi2s_root / "artifacts/intermediate_checkpoint_hashes.json"
    verify_sha(
        intermediate_manifest_path,
        INTERMEDIATE_HASH_MANIFEST_SHA,
        "intermediate checkpoint hash manifest",
    )
    intermediate_manifest = json.loads(intermediate_manifest_path.read_text())
    hashed_intermediates = {row["path"]: row for row in intermediate_manifest["checkpoints"]}
    intermediates = []
    for model, seed in (("B_HVA", 42), ("B_HVA", 43)):
        final = Path(
            checkpoint_by_key[(model, seed)]["path"].replace("$EXPERIMENT_ROOT", str(experiments))
        )
        for step in (10000, 20000):
            path = final.parent / str(step)
            if not (path / "params/manifest.ocdbt").is_file():
                raise RuntimeError(f"missing intermediate checkpoint: {path}")
            symbolic_path = symbolic(path, experiments)
            hashed = hashed_intermediates.get(symbolic_path)
            if hashed is None:
                raise RuntimeError(f"intermediate checkpoint lacks content hash: {symbolic_path}")
            intermediates.append(
                {
                    "model": model,
                    "training_seed": seed,
                    "directory_step": step,
                    "path": symbolic_path,
                    "tree_sha256": hashed["tree_sha256"],
                    "files": int(hashed["files"]),
                    "bytes": int(hashed["bytes"]),
                    "identity_source": "$PI2S_ROOT/artifacts/intermediate_checkpoint_hashes.json",
                    "identity_source_sha256": INTERMEDIATE_HASH_MANIFEST_SHA,
                    "closed_loop_authorized": False,
                    "identity_status": "FULL_CONTENT_HASH_VERIFIED",
                }
            )

    contact_path = (
        ROOT / ".local/datasets/simulation/s4_3_pi1/pinch_tongs_official_tactile/sidecar.npz"
    )
    va27_path = ROOT / ".local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
    contact_snapshot = pi2s_root / "source_snapshots/data/pi1_contact_sidecar.npz"
    va27_snapshot = pi2s_root / "source_snapshots/data/pi2m_va27_sidecar.npz"
    if sha256_file(contact_path) != CONTACT_SHA or sha256_file(va27_path) != VA27_SHA:
        raise RuntimeError("sidecar identity mismatch")
    if sha256_file(contact_snapshot) != CONTACT_SHA or sha256_file(va27_snapshot) != VA27_SHA:
        raise RuntimeError("persistent sidecar snapshot identity mismatch")
    official_manifest_path = (
        ROOT / ".local/artifacts/simulation/s4_3_pi0/official_dataset_manifest.json"
    )
    raw_manifest_path = ROOT / ".local/artifacts/simulation/s4_3_pi1/raw_dataset_manifest.json"
    alignment_path = ROOT / ".local/artifacts/simulation/s4_3_pi1/official_dataset_alignment.json"
    policy_membership_path = (
        ROOT / ".local/artifacts/simulation/s4_3_pd/policy_expert_dataset_manifest.json"
    )
    policy_cache_manifest_path = ROOT / ".local/cache/simulation/s4_3_restart/manifest.json"
    for path, expected, label in (
        (official_manifest_path, OFFICIAL_DATASET_MANIFEST_SHA, "official dataset manifest"),
        (raw_manifest_path, RAW_DATASET_MANIFEST_SHA, "raw dataset manifest"),
        (alignment_path, ALIGNMENT_MANIFEST_SHA, "official/raw alignment"),
        (policy_membership_path, POLICY_DEV_MANIFEST_SHA, "POLICY_DEV membership"),
        (policy_cache_manifest_path, POLICY_DEV_CACHE_MANIFEST_SHA, "POLICY_DEV cache manifest"),
    ):
        verify_sha(path, expected, label)
    raw_manifest = json.loads(raw_manifest_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    policy_membership = json.loads(policy_membership_path.read_text())
    policy_cache_manifest = json.loads(policy_cache_manifest_path.read_text())
    source_groups = alignment["raw_episode_names_in_converter_sort_order"]
    episode_lengths = [int(value) for value in alignment["episode_lengths"]]
    raw_trim = [
        int(value)
        for value in alignment["raw_leading_static_frames_excluded_by_official_conversion"]
    ]
    if (
        len(source_groups) != 100
        or len(set(source_groups)) != 100
        or len(episode_lengths) != 100
        or len(raw_trim) != 100
        or raw_manifest["episode_names"] != source_groups
    ):
        raise RuntimeError("TRAIN source-group mapping identity mismatch")
    train_group_mapping = [
        {
            "episode_index": index,
            "source_group_id": source_groups[index],
            "converted_length": episode_lengths[index],
            "raw_leading_trim": raw_trim[index],
            "raw_length": int(alignment["raw_episode_lengths"][index]),
        }
        for index in range(100)
    ]
    train_group_mapping_sha = canonical_sha(train_group_mapping)
    with (
        np.load(contact_path, allow_pickle=False) as contact,
        np.load(va27_path, allow_pickle=False) as va27,
    ):
        required = {
            "index",
            "episode_index",
            "frame_index",
            "tactile_sim",
            "tactile_history",
            "history_bootstrap_count",
            "control_tick_end",
            "contact_state",
            "contact_shared_target",
            "physical_aux_valid",
        }
        if set(contact.files) != required:
            raise RuntimeError(f"unexpected contact fields: {set(contact.files)}")
        if contact["tactile_history"].shape != (40065, 26, 30):
            raise RuntimeError("history shape mismatch")
        if contact["contact_state"].shape != (40065, 256):
            raise RuntimeError("H shape mismatch")
        np.testing.assert_array_equal(contact["index"], va27["index"])
        np.testing.assert_array_equal(contact["physical_aux_valid"], va27["va_aux_valid"])
        valid = contact["physical_aux_valid"].astype(bool)
        if int(valid.sum()) != 37365 or int((~valid).sum()) != 2700:
            raise RuntimeError("tail mask counts mismatch")
        selected, keys, train_identities = select_train_rows(
            contact["episode_index"],
            contact["frame_index"],
            contact["control_tick_end"],
            valid,
            episode_lengths,
            raw_trim,
            source_groups,
        )
        fixed = select_balanced(selected, keys, contact["tactile_sim"], 16)
        fixed_labels = {
            str(row): (
                "active"
                if np.linalg.norm(contact["tactile_sim"][row].astype(np.float64)) > 0.0
                else "inactive"
            )
            for row in fixed
        }
        gradient_rows = select_balanced(selected, keys, contact["tactile_sim"], 4)
        active_gradient = [
            row
            for row in gradient_rows
            if np.linalg.norm(contact["tactile_sim"][row].astype(np.float64)) > 0.0
        ]
        inactive_gradient = [row for row in gradient_rows if row not in active_gradient]
        gradient_batches = [
            [active_gradient[index], inactive_gradient[index]] for index in range(4)
        ]
        episode_values = {row: int(contact["episode_index"][row]) for row in fixed}
        fixed_by_key = sorted(fixed, key=keys.__getitem__)
        other_episode_map = {}
        for position, row in enumerate(fixed_by_key):
            for offset in range(1, len(fixed_by_key) + 1):
                candidate = fixed_by_key[(position + offset) % len(fixed_by_key)]
                if episode_values[candidate] != episode_values[row]:
                    other_episode_map[str(row)] = candidate
                    break
        if len(other_episode_map) != len(fixed):
            raise RuntimeError("other-episode mapping incomplete")

        episode_rows: dict[int, list[int]] = {}
        for row in range(contact["episode_index"].shape[0]):
            episode_rows.setdefault(int(contact["episode_index"][row]), []).append(row)
        for rows in episode_rows.values():
            rows.sort(key=lambda row: (int(contact["control_tick_end"][row]), row))
        lag5_map = {}
        for row in fixed_by_key:
            current_tick = int(contact["control_tick_end"][row])
            rows = episode_rows[int(contact["episode_index"][row])]
            requested_tick = current_tick - 5
            eligible = [
                candidate
                for candidate in rows
                if int(contact["control_tick_end"][candidate]) <= requested_tick
            ]
            candidate = eligible[-1] if eligible else rows[0]
            actual_tick = int(contact["control_tick_end"][candidate])
            bootstrap_affected = actual_tick != requested_tick
            if not bootstrap_affected and current_tick - actual_tick != 5:
                raise RuntimeError(f"lag5 control-tick invariant failed at row {row}")
            lag5_map[str(row)] = {
                "row": candidate,
                "episode_index": int(contact["episode_index"][row]),
                "current_frame": int(contact["frame_index"][row]),
                "selected_frame": int(contact["frame_index"][candidate]),
                "current_control_tick": current_tick,
                "requested_control_tick": requested_tick,
                "actual_control_tick": actual_tick,
                "bootstrap_affected": bootstrap_affected,
            }
        train_mean_h = np.mean(contact["contact_state"].astype(np.float64), axis=0).astype(
            np.float32
        )
        train_mean_h_sha = hashlib.sha256(train_mean_h.tobytes(order="C")).hexdigest()
        train_mean_h_l2 = float(np.linalg.norm(train_mean_h.astype(np.float64)))

    policy_cache_root = ROOT / ".local/cache/simulation/s4_3_restart/pinch_tongs/dev"
    policy_cache_entry = policy_cache_manifest["tasks"]["pinch_tongs"]["dev"]
    for field, expected_sha in policy_cache_entry["sha256"].items():
        suffix = ".json" if field == "episodes" else ".npy"
        verify_sha(policy_cache_root / f"{field}{suffix}", expected_sha, f"POLICY_DEV {field}")
    policy_selected, policy_keys, policy_identities, policy_episode_ids = select_policy_dev_rows(
        policy_cache_root, policy_membership
    )
    policy_selected_records = [policy_identities[row] for row in policy_selected]
    policy_selected_groups = {identity["source_group_id"] for identity in policy_selected_records}
    policy_selected_attempts = {identity["attempt_id"] for identity in policy_selected_records}
    if len(policy_selected_groups) != 5 or len(policy_selected_attempts) != 25:
        raise RuntimeError("POLICY_DEV selection did not cover all five groups and 25 attempts")
    policy_overlap_indices = sorted(source_groups.index(group) for group in policy_selected_groups)
    if policy_overlap_indices != [24, 25, 26, 27, 28]:
        raise RuntimeError(f"unexpected original pi0.5 TRAIN overlap: {policy_overlap_indices}")
    train_selection_records = [train_identities[row] for row in selected]
    snapshot_root = pi2s_root / f"source_snapshots/{INPUT_SNAPSHOT_VERSION}"
    if prepare_inputs:
        snapshot_manifest = prepare_input_snapshot(
            pi2s_root=pi2s_root,
            contact_path=contact_path,
            va27_path=va27_path,
            train_selected=selected,
            train_identities=train_identities,
            fixed_rows=fixed_by_key,
            policy_cache_root=policy_cache_root,
            policy_selected=policy_selected,
            policy_identities=policy_identities,
            policy_episode_ids=policy_episode_ids,
            policy_membership=policy_membership,
        )
    else:
        snapshot_manifest = verify_input_snapshot(
            snapshot_root,
            canonical_sha(train_selection_records),
            canonical_sha(policy_selected_records),
        )

    dev_manifest_path = ROOT / ".local/artifacts/simulation/s4_3_pi2n/development_manifest.json"
    dev_manifest_snapshot = pi2s_root / "source_snapshots/data/pi2n_development_manifest.json"
    if (
        sha256_file(dev_manifest_path) != DEV_MANIFEST_SHA
        or sha256_file(dev_manifest_snapshot) != DEV_MANIFEST_SHA
    ):
        raise RuntimeError("development reset manifest identity mismatch")
    development = json.loads(dev_manifest_path.read_text())
    reset_specs = sorted(development["reset_specs"], key=lambda row: row["reset_identity"])
    selected_resets = reset_specs[:6]
    for relative, expected_sha in SOURCE_IDENTITIES.items():
        verify_sha(ROOT / relative, expected_sha, relative)
    et_checkpoint = experiments / "simulation/s4_2r/contact_state/accepted.pt"
    verify_sha(et_checkpoint, ET_CHECKPOINT_SHA, "accepted E_T checkpoint")
    openpi_client_root = (
        ROOT / ".local/external/simulation/s4_3_pi1/openpi/packages/openpi-client/src/openpi_client"
    )
    verify_sha(
        openpi_client_root / "__init__.py",
        OPENPI_CLIENT_INIT_SHA,
        "accepted OpenPI client package initializer",
    )
    verify_sha(
        openpi_client_root / "image_tools.py",
        OPENPI_CLIENT_IMAGE_TOOLS_SHA,
        "accepted OpenPI client image preprocessing",
    )
    client_files, client_tree_sha = python_tree_sha(openpi_client_root)
    if (
        client_files != OPENPI_CLIENT_PYTHON_FILES
        or client_tree_sha != OPENPI_CLIENT_PYTHON_TREE_SHA
    ):
        raise RuntimeError("accepted OpenPI client Python tree identity mismatch")
    tokenizer_snapshot = pi2s_root / PALIGEMMA_TOKENIZER_SNAPSHOT
    verify_sha(
        tokenizer_snapshot,
        PALIGEMMA_TOKENIZER_SHA,
        "persistent PaliGemma tokenizer asset",
    )
    if tokenizer_snapshot.stat().st_size != PALIGEMMA_TOKENIZER_BYTES:
        raise RuntimeError("persistent PaliGemma tokenizer byte count mismatch")
    tokenizer_manifest = pi2s_root / PALIGEMMA_TOKENIZER_MANIFEST
    verify_sha(
        tokenizer_manifest,
        PALIGEMMA_TOKENIZER_MANIFEST_SHA,
        "persistent PaliGemma tokenizer manifest",
    )

    protocol: dict[str, Any] = {
        "schema": "tactile3d-unit.s4-3-pi2s-diagnostic-protocol.v1",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "protocol_parent_commit": BASE_HEAD,
        "diagnostic_script_sha256": sha256_file(Path(__file__)),
        "diagnostic_implementation_sha256": {
            name: sha256_file(ROOT / relative_path)
            for name, relative_path in sorted(DIAGNOSTIC_IMPLEMENTATION_PATHS.items())
        },
        "scientific_role": "POST_HOC_DEVELOPMENT_DIAGNOSIS",
        "formal_track_a_outcomes_read_only": 3000,
        "training_budget": {"teacher_runs": 0, "policy_runs": 0, "optimizer_updates": 0},
        "all_final_checkpoints": all_checkpoints,
        "focus_checkpoints": focus,
        "intermediate_checkpoints": intermediates,
        "source_manifests": {
            "official_dataset": {
                "path": symbolic_repo(official_manifest_path),
                "sha256": OFFICIAL_DATASET_MANIFEST_SHA,
            },
            "raw_dataset": {
                "path": symbolic_repo(raw_manifest_path),
                "sha256": RAW_DATASET_MANIFEST_SHA,
                "raw_cache_sha256": raw_manifest["cache_sha256"],
                "raw_revision": raw_manifest["revision"],
            },
            "official_raw_alignment": {
                "path": symbolic_repo(alignment_path),
                "sha256": ALIGNMENT_MANIFEST_SHA,
                "official_parquet_sha256": alignment["official_parquet_sha256"],
            },
            "policy_dev_membership": {
                "path": symbolic_repo(policy_membership_path),
                "sha256": POLICY_DEV_MANIFEST_SHA,
            },
            "policy_dev_cache": {
                "path": symbolic_repo(policy_cache_manifest_path),
                "sha256": POLICY_DEV_CACHE_MANIFEST_SHA,
                "content_sha256": policy_cache_manifest["cache_content_sha256"],
            },
            "persistent_input_snapshot": {
                "path": f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}",
                "manifest_sha256": sha256_file(snapshot_root / "manifest.json"),
                "files": len(snapshot_manifest["files"]),
                "bytes": sum(row["bytes"] for row in snapshot_manifest["files"]),
                "status": snapshot_manifest["status"],
            },
        },
        "data_contract": {
            "contact_sidecar": symbolic_repo(contact_path),
            "contact_sidecar_sha256": CONTACT_SHA,
            "contact_sidecar_persistent_snapshot": "$PI2S_ROOT/source_snapshots/data/pi1_contact_sidecar.npz",
            "va27_sidecar": symbolic_repo(va27_path),
            "va27_sidecar_sha256": VA27_SHA,
            "va27_sidecar_persistent_snapshot": "$PI2S_ROOT/source_snapshots/data/pi2m_va27_sidecar.npz",
            "rows": 40065,
            "valid_plus27_targets": 37365,
            "tail_invalid_rows": 2700,
            "raw_control_hz": 50,
            "history_ticks": 26,
            "history_seconds_approx": 0.5,
            "target_ticks": 27,
            "target_seconds": 0.54,
            "legacy_bva_ticks": 16,
            "legacy_bva_seconds": 0.32,
            "source_group_mapping": (
                "episode_index -> official_raw_alignment.raw_episode_names_in_converter_sort_order"
            ),
            "source_group_mapping_sha256": train_group_mapping_sha,
            "source_groups": 100,
            "source_group_mapping_status": "AVAILABLE_AND_VERIFIED",
            "development_reset_manifest": symbolic_repo(dev_manifest_path),
            "development_reset_manifest_persistent_snapshot": (
                "$PI2S_ROOT/source_snapshots/data/pi2n_development_manifest.json"
            ),
            "development_reset_manifest_sha256": DEV_MANIFEST_SHA,
            "official_converter_fps_label": 30,
            "converter_label_warning": (
                "30 Hz is metadata used by the loader; alignment proves row order/trim, not a "
                "physical 50-to-30 Hz resampling operation"
            ),
            "fixed_train_observation_snapshot": (
                f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed"
            ),
            "policy_dev_snapshot": (
                f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/policy_dev"
            ),
            "policy_dev_rows_available": 8815,
            "policy_dev_source_groups_available": 5,
            "policy_dev_attempts_available": 25,
            "policy_dev_original_pi05_train_overlap": {
                "status": "OVERLAPS_ORIGINAL_POLICY_TRAIN",
                "official_train_episode_indices": policy_overlap_indices,
                "interpretation": (
                    "isolated from the later POLICY_EXPERT_TRAIN split only; not an unseen source-group "
                    "holdout from the original pi0.5 training corpus"
                ),
            },
        },
        "runtime_dependencies": {
            "openpi_client": {
                "accepted_package_initializer": (
                    "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                    "openpi-client/src/openpi_client/__init__.py"
                ),
                "accepted_package_initializer_sha256": OPENPI_CLIENT_INIT_SHA,
                "accepted_image_tools": (
                    "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                    "openpi-client/src/openpi_client/image_tools.py"
                ),
                "accepted_image_tools_sha256": OPENPI_CLIENT_IMAGE_TOOLS_SHA,
                "accepted_python_files": OPENPI_CLIENT_PYTHON_FILES,
                "accepted_python_tree_sha256": OPENPI_CLIENT_PYTHON_TREE_SHA,
                "required_distribution_version": "0.1.0",
                "runtime_origin_policy": (
                    "REPOSITORY_SCOPED_AND_BYTE_IDENTICAL_TO_ACCEPTED_SOURCE"
                ),
            },
            "paligemma_tokenizer": {
                "upstream_uri": "gs://big_vision/paligemma_tokenizer.model",
                "persistent_snapshot": (f"$PI2S_ROOT/{PALIGEMMA_TOKENIZER_SNAPSHOT}"),
                "persistent_manifest": (f"$PI2S_ROOT/{PALIGEMMA_TOKENIZER_MANIFEST}"),
                "persistent_manifest_sha256": PALIGEMMA_TOKENIZER_MANIFEST_SHA,
                "sha256": PALIGEMMA_TOKENIZER_SHA,
                "bytes": PALIGEMMA_TOKENIZER_BYTES,
                "required_sentencepiece_version": "0.2.2",
                "execution_cache_policy": (
                    "COPY_VERIFIED_SNAPSHOT_TO_EPHEMERAL_OPENPI_DATA_HOME_NO_NETWORK"
                ),
            },
            "execution_environment": {
                "pip_freeze_all_sha256": OPENPI_ENVIRONMENT_FREEZE_SHA,
                "identity_source": "historical starting_integrity environment freeze",
            },
        },
        "samples": {
            "train_windows": selected,
            "train_windows_count": len(selected),
            "train_window_records": train_selection_records,
            "train_window_records_sha256": canonical_sha(train_selection_records),
            "train_identity_keys_sha256": canonical_sha([keys[row] for row in selected]),
            "train_source_groups_covered": len(
                {train_identities[row]["source_group_id"] for row in selected}
            ),
            "train_selection": (
                "canonical SHA256 of frozen identity; one minimum-hash full-action30 row per true "
                "source_group, then global identity-hash fill; no value or outcome label in key"
            ),
            "train_candidate_contract": (
                "physical_aux_valid and official action horizon t..t+29 within episode; H26/current/+27 "
                "target/full action30 share support"
            ),
            "policy_dev_windows": policy_selected_records,
            "policy_dev_windows_count": len(policy_selected),
            "policy_dev_window_records_sha256": canonical_sha(policy_selected_records),
            "policy_dev_identity_keys_sha256": canonical_sha(
                [policy_keys[row] for row in policy_selected]
            ),
            "policy_dev_source_groups_covered": len(policy_selected_groups),
            "policy_dev_attempts_covered": len(policy_selected_attempts),
            "policy_dev_status": "AVAILABLE_FOR_H_ACTION_DOMAIN_DISTRIBUTION_ONLY",
            "policy_dev_parent_cohort": "SUCCESS_ONLY_NATIVE_SUCCESS_EXPERT_TRAJECTORIES",
            "policy_dev_row_selection_scope": (
                "OUTCOME_BLIND_ONLY_WITHIN_THE_PREEXISTING_SUCCESS_CONDITIONED_PARENT_COHORT"
            ),
            "policy_dev_population_or_failure_comparison_allowed": False,
            "policy_dev_fixed_forward_compatibility": (
                "NOT_COMPATIBLE_WITH_OFFICIAL_PI05_OBSERVATION_CONTRACT"
            ),
            "policy_dev_missing_official_fields": [
                "paired front+wrist cameras",
                "state[23]",
                "task prompt field",
                "action horizon[30,22]",
            ],
            "policy_dev_selection": (
                "canonical SHA256 of membership/cache/source/attempt/tick identity; one minimum-hash row "
                "per source group then global fill; per-attempt steps content SHA remains provenance "
                "but is explicitly excluded from the selection-key projection; no tensor value or "
                "success/failure label in key"
            ),
            "h_parity_rows": selected[:64],
            "h_parity_rows_sha256": canonical_sha(selected[:64]),
            "fixed_observation_rows": fixed_by_key,
            "fixed_observation_contact_strata": {"active": 16, "inactive": 16},
            "fixed_observation_contact_labels": fixed_labels,
            "contact_stratum_definition": "active iff float64 L2(tactile_sim[row]) > 0",
            "gradient_batches": gradient_batches,
            "gradient_minibatches_per_checkpoint": 4,
            "gradient_batch_size": 2,
        },
        "encoder_contract": {
            "accepted_et_checkpoint": symbolic(et_checkpoint, experiments),
            "accepted_et_checkpoint_sha256": ET_CHECKPOINT_SHA,
            "input_shape": [26, 30],
            "output_shape": [256],
            "history_policy": "LEFT_REPEAT_FIRST within episode; reset before every episode",
            "paths": {
                "shared_teacher_loader": {
                    "implementation": "$REPO_ROOT/gr00t/simulation/sim_contact_models.py",
                    "implementation_sha256": SOURCE_IDENTITIES[
                        "gr00t/simulation/sim_contact_models.py"
                    ],
                    "architecture_implementation": "$REPO_ROOT/gr00t/tactile_teacher/models.py",
                    "architecture_implementation_sha256": SOURCE_IDENTITIES[
                        "gr00t/tactile_teacher/models.py"
                    ],
                },
                "cached_training_sidecar": {
                    "implementation": "$REPO_ROOT/scripts/simulation/build_s4_3_pi1_sidecar.py",
                    "implementation_sha256": SOURCE_IDENTITIES[
                        "scripts/simulation/build_s4_3_pi1_sidecar.py"
                    ],
                    "stored_output": "$REPO_ROOT/.local/datasets/simulation/s4_3_pi1/"
                    "pinch_tongs_official_tactile/sidecar.npz",
                    "stored_output_sha256": CONTACT_SHA,
                },
                "direct_et": {
                    "implementation": "$REPO_ROOT/gr00t/simulation/s4_3_act.py",
                    "implementation_sha256": SOURCE_IDENTITIES["gr00t/simulation/s4_3_act.py"],
                    "entrypoint": "FrozenS42PolicyStack.encode_contact_state",
                },
                "live_sidecar": {
                    "runtime_implementation": "$REPO_ROOT/gr00t/simulation/pi1d_runtime.py",
                    "runtime_implementation_sha256": SOURCE_IDENTITIES[
                        "gr00t/simulation/pi1d_runtime.py"
                    ],
                    "history_implementation": "$REPO_ROOT/gr00t/simulation/s4_3_pi1.py",
                    "history_implementation_sha256": SOURCE_IDENTITIES[
                        "gr00t/simulation/s4_3_pi1.py"
                    ],
                    "service_implementation": (
                        "$REPO_ROOT/scripts/simulation/serve_s4_3_pi1_contact_state.py"
                    ),
                    "service_implementation_sha256": SOURCE_IDENTITIES[
                        "scripts/simulation/serve_s4_3_pi1_contact_state.py"
                    ],
                    "entrypoints": [
                        "CausalContactRuntime.reset/append/contact_state",
                        "ContactStateUnixClient.encode",
                    ],
                },
            },
            "model_integration_source": {
                "path": "$REPO_ROOT/gr00t/simulation/pi05_tactile_unit.py",
                "sha256": SOURCE_IDENTITIES["gr00t/simulation/pi05_tactile_unit.py"],
            },
        },
        "h_conditions": {
            "enabled_models": ["B1", "B_HVA", "B2"],
            "conditions": ["correct", "train_mean", "same_episode_lag5", "other_episode", "zero"],
            "train_mean_definition": (
                "float64 mean over all 40065 frozen TRAIN contact_state rows, cast to float32"
            ),
            "train_mean_h_float32_sha256": train_mean_h_sha,
            "train_mean_h_l2": train_mean_h_l2,
            "lag_bootstrap": "clamp_to_first_episode_row_and_mark_affected",
            "same_episode_lag5_row_map": lag5_map,
            "other_episode_row_map": other_episode_map,
            "zero_label": "OUT_OF_DISTRIBUTION_NUMERICAL_DIAGNOSTIC",
        },
        "prefix_integration_contract": {
            "contact_state_shape": [256],
            "adapter_output_shape_per_observation": [8, 2048],
            "token_count": 8,
            "token_width": 2048,
            "placement": "APPEND_AFTER_UNCHANGED_OFFICIAL_PREFIX",
            "token_order": "CONTACT_ADAPTER_RESHAPE_ORDER_0_THROUGH_7",
            "input_mask_for_added_tokens": True,
            "autoregressive_mask_for_added_tokens": False,
            "position_index_rule": "cumsum(input_mask)-1",
            "padding_interaction": (
                "physical append occurs after the padded official prefix; masked padding does not "
                "advance the added tokens' position indices"
            ),
            "state_dimension_changed": False,
            "action_dimension_changed": False,
        },
        "fixed_sampling": {
            "noise_distribution": "jax.random.normal",
            "noise_shape": [30, 32],
            "noise_dtype": "float32",
            "policy_output_shape": [30, 22],
            "noise_seeds": list(range(431700, 431732)),
            "shared_noise_across_h_conditions": True,
            "observation_source": (
                f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed"
            ),
            "official_observation_fields": [
                "front RGB CHW uint8 losslessly decoded from converter video",
                "wrist RGB CHW uint8 losslessly decoded from converter video",
                "state[23]",
                "task prompt",
            ],
            "normalization_source": "each frozen checkpoint assets/local_repo/norm_stats.json",
            "policy_dev_observations_used": False,
            "policy_dev_exclusion_reason": (
                "single random camera/proprio22/no prompt cannot be silently mapped to official "
                "front+wrist/state23/prompt"
            ),
        },
        "gradient_diagnostic": {
            "flow_times": [0.1, 0.5, 0.9],
            "flow_time_semantics": "x_t = time * noise + (1 - time) * action; target = noise - action",
            "nuisance_noise_seeds_by_minibatch": [431900, 431901, 431902, 431903],
            "nuisance_noise_distribution": "jax.random.normal",
            "nuisance_noise_shape_per_minibatch": [2, 30, 32],
            "nuisance_noise_dtype": "float32",
            "shared_batch_and_noise_across_checkpoints_and_flow_times": True,
            "batch_sources": {
                "official_observation_and_action": (
                    f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed/"
                    "{front_rgb_chw_uint8,wrist_rgb_chw_uint8,state,action,action_is_pad}.npy"
                ),
                "contact_history_and_h": (
                    f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed/"
                    "{tactile_history,contact_state}.npy"
                ),
                "contact_auxiliary": (
                    f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed/"
                    "{contact_target_t_plus_27,contact_aux_valid}.npy"
                ),
                "va_auxiliary": (
                    f"$PI2S_ROOT/source_snapshots/{INPUT_SNAPSHOT_VERSION}/train_fixed/"
                    "{va_target_t_plus_27,va_aux_valid}.npy"
                ),
                "va27_full_source_identity": {
                    "path": "$PI2S_ROOT/source_snapshots/data/pi2m_va27_sidecar.npz",
                    "sha256": VA27_SHA,
                    "join": "same frozen global row index; index equality verified before snapshot",
                },
            },
            "main_loss": (
                "mean official flow-matching squared error over batch, 30 action positions, "
                "and internal action_dim32 at frozen explicit time/noise"
            ),
            "auxiliary_loss_reduction": (
                "per-sample mean squared error over target[8,32], reduced as "
                "sum(sample_loss * model-specific valid mask) / max(sum(valid),1)"
            ),
            "auxiliary_loss_by_model": {
                "B0": "N/A_NO_AUXILIARY_OBJECTIVE",
                "B1": "N/A_NO_AUXILIARY_OBJECTIVE",
                "B_VA27": {
                    "target": "va_target_t_plus_27.npy (source field va_shared_target)",
                    "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
                },
                "B_HVA": {
                    "target": "va_target_t_plus_27.npy (source field va_shared_target)",
                    "valid_mask": "va_aux_valid.npy (source field va_aux_valid)",
                },
                "B2": {
                    "target": "contact_target_t_plus_27.npy (source field contact_shared_target)",
                    "valid_mask": "contact_aux_valid.npy (source field physical_aux_valid)",
                },
            },
            "combined_loss": "main_loss + checkpoint-recipe frozen lambda * auxiliary_loss",
            "parameter_membership": (
                "enumerate actual leaves accepted by the reconstructed config.trainable_filter; "
                "do not infer membership from log metric names"
            ),
            "main_auxiliary_cosine_scope": (
                "all_common_trainable is the structurally shared config-trainable path set with "
                "the objective-specific physical_auxiliary and action_out_proj heads explicitly "
                "excluded; each cosine uses only the finite non-None g_main/g_aux leaf intersection, "
                "with lora, contact_adapter, and physical_auxiliary also reported separately and "
                "empty/zero intersections marked N/A"
            ),
            "weighted_auxiliary_to_main_ratio": (
                "lambda * L_aux / max(abs(L_main), float64 tiny), reported descriptively"
            ),
            "parameter_groups": [
                "all_common_trainable",
                "lora",
                "contact_adapter",
                "physical_auxiliary",
            ],
            "invalid_tail_policy": "physical auxiliary strictly masked; fixed rows have full action30/+27 support",
            "optimizer_step": False,
            "state_mutation_allowed": False,
            "required_before_after_hash_match": True,
            "no_rng_or_buffer_mutation": True,
        },
        "numeric_tolerances": {
            "float32_same_path_atol": 1e-6,
            "float32_same_path_rtol": 1e-5,
            "finite_required": True,
            "source": "PRE_RESULT_ENGINEERING_TOLERANCE_NO_CLOSED_LOOP_OUTCOME_CONSULTED",
            "determination": (
                "frozen before the 64-row formal audit; conservative float32 same-model-path "
                "allclose contract following a one-history direct/live engineering smoke that "
                "was bitwise equal; the formal report still records bitwise flags and the full "
                "predeclared absolute-error quantiles"
            ),
            "post_hoc_relaxation_allowed": False,
        },
        "metrics": {
            "h_parity": {
                "comparison_dtype": "float32",
                "absolute_error_quantile_levels": [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
                "reductions": [
                    "array_equal bitwise flag per pair and aggregate",
                    "max absolute error over all selected rows/elements",
                    "mean absolute error over all selected rows/elements",
                    "listed quantiles over flattened absolute errors",
                    "shape and finite fraction",
                ],
                "pairs": ["cached_vs_direct", "cached_vs_live", "direct_vs_live"],
            },
            "prefix_contract": [
                "official_prefix_unchanged",
                "added_token_count",
                "input_mask",
                "autoregressive_mask",
                "position_indices_append_after_official_prefix",
                "contact_state_pre_layernorm_l2",
                "contact_state_post_layernorm_l2",
                "image_token_l2_at_width_2048",
                "language_token_l2_at_width_2048",
                "contact_token_l2_at_width_2048",
            ],
            "fixed_observation_actions": {
                "reference": "each intervention minus same-row correct-H output under identical noise",
                "delta_shape": [30, 22],
                "component_rms": {
                    "tcp_xyz": "sqrt(mean(delta[:,0:3]^2))",
                    "rotation_vector": "sqrt(mean(delta[:,3:6]^2))",
                    "hand": "sqrt(mean(delta[:,6:22]^2))",
                },
                "chunk_l2": {
                    "per_step": "L2 over output22",
                    "early": "mean of per-step L2 for positions 0:10",
                    "late": "mean of per-step L2 for positions 20:30",
                },
                "jerk_proxy": (
                    "mean L2 of second finite difference over chunk positions in raw policy action "
                    "units per control-tick^2; descriptive proxy, not physical jerk"
                ),
                "runtime_clipped_fraction": "N/A_NO_EXPLICIT_RUNTIME_ACTION_CLIP_IN_POLICY_INFER",
                "normalization_boundary_exceedance": (
                    "fraction below checkpoint norm q01 or above q99, by xyz/rotvec/hand; not clipping"
                ),
                "aggregation": "report per row and mean/median/q10/q90 over the 32 frozen rows",
                "post_hoc_threshold": None,
            },
            "gradients": {
                "losses": ["main_loss", "auxiliary_loss", "combined_loss"],
                "per_group": ["l2_norm_g_main", "l2_norm_g_aux", "cosine_g_main_g_aux"],
                "flow_times": [0.1, 0.5, 0.9],
                "aggregation": "retain all checkpoint/minibatch/time records; summarize median and range only",
                "b0_b1_auxiliary": "N/A_NO_AUXILIARY_OBJECTIVE",
                "post_hoc_threshold": None,
                "preprocess_mode": "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION",
                "training_equivalence_scope": (
                    "same frozen model/loss/mask/target path with explicit time and noise; "
                    "does not reproduce stochastic train=True image augmentation"
                ),
            },
            "interpretation": "descriptive diagnostics; no post-hoc PASS threshold or causal claim from one proxy",
        },
        "historical_stage_observability": {
            "object_contact_ever": "AVAILABLE_EPISODE_AGGREGATE",
            "max_native_pinch_count": "AVAILABLE_EPISODE_AGGREGATE",
            "native_success": "AVAILABLE",
            "approach": "NOT_AVAILABLE",
            "stable_grasp": "NOT_AVAILABLE",
            "lift": "NOT_AVAILABLE",
            "exact_first_failure_step": "NOT_AVAILABLE",
            "rule": "DO_NOT_INFER_MISSING_STAGES_FROM_FINAL_FRAME_OR_VIDEO",
        },
        "development_rollouts": {
            "base": {
                "focus_checkpoints": 6,
                "reset_specs": selected_resets,
                "sampling_seeds": [4317, 4318],
                "canonical_tuples": 72,
                "reset_selection": (
                    "six lexicographically smallest reset_identity values from frozen 30-reset "
                    "PI2N development manifest; no performance screening"
                ),
            },
            "optional_h_intervention": {
                "enabled_only_after_written_s2_s3_information_need": True,
                "models": ["B_HVA_42", "B_HVA_43", "B1_43"],
                "reset_specs": selected_resets[:4],
                "conditions": ["train_mean", "same_episode_lag5"],
                "sampling_seeds": [4317, 4318],
                "canonical_tuples": 48,
            },
            "maximum_canonical_tuples": 120,
            "infrastructure_retry_per_tuple": 1,
            "performance_selection_allowed": False,
            "formal_success_rate_replacement_allowed": False,
        },
        "missing_inputs": {
            "attention_weights": "NOT_AVAILABLE_NO_INTRUSIVE_MODEL_CHANGE_AUTHORIZED",
            "historical_step_telemetry": "NOT_AVAILABLE",
            "policy_dev_official_pi05_fixed_observations": (
                "NOT_AVAILABLE_SINGLE_RANDOM_CAMERA_PROPRIO22_NO_PROMPT; raw windows retained for "
                "H/action/domain diagnostics without fabricating missing fields"
            ),
        },
        "completion_policy": {
            "unique_root_cause_required": False,
            "inconclusive_valid": True,
            "automatic_budget_expansion": False,
            "automatic_retraining": False,
            "automatic_teacher_replacement": False,
        },
        "selection_rationale": {
            "focus_checkpoints": (
                "predeclared structural contrasts B0_43, B_VA27_43, B1_43, B_HVA_42, "
                "B_HVA_43, and B2_43; known formal results are not used to expand the set"
            ),
            "offline_windows": (
                "identity hash order with one row per verified source group before global fill; "
                "keys exclude tensor values and formal rollout success/failure labels; POLICY_DEV "
                "row selection is outcome-blind only within its preexisting success-only expert "
                "parent cohort and cannot support a population or failure comparison"
            ),
        },
    }
    protocol["selection_sha256"] = canonical_sha(
        {
            "train_records": train_selection_records,
            "policy_dev_records": policy_selected_records,
            "fixed": fixed_by_key,
            "gradient": gradient_batches,
            "resets": selected_resets,
            "noise": protocol["fixed_sampling"]["noise_seeds"],
            "gradient_noise": protocol["gradient_diagnostic"]["nuisance_noise_seeds_by_minibatch"],
            "train_mean_h": train_mean_h_sha,
            "lag5": lag5_map,
            "other_episode": other_episode_map,
        }
    )
    return protocol


def validate(protocol: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    experiments = (ROOT / ".local/experiments").resolve(strict=True)
    pi2s_root = (experiments / "simulation/s4_3_pi2s").resolve(strict=True)

    def require(condition: bool, name: str) -> None:
        if not condition:
            raise RuntimeError(f"protocol check failed: {name}")
        checks.append(name)

    def is_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    require(protocol["status"] == "PREREGISTERED_NOT_EXECUTED", "not executed")
    require(
        protocol.get("diagnostic_script_sha256") == sha256_file(Path(__file__)),
        "protocol generator byte identity",
    )
    implementation_hashes = protocol.get("diagnostic_implementation_sha256")
    require(isinstance(implementation_hashes, dict), "diagnostic implementation hash mapping")
    require(
        set(implementation_hashes) == set(DIAGNOSTIC_IMPLEMENTATION_PATHS),
        "diagnostic implementation hash key set",
    )
    for name, relative_path in sorted(DIAGNOSTIC_IMPLEMENTATION_PATHS.items()):
        require(
            implementation_hashes[name] == sha256_file(ROOT / relative_path),
            f"diagnostic implementation byte identity: {name}",
        )
    runtime_dependencies = protocol.get("runtime_dependencies")
    require(isinstance(runtime_dependencies, dict), "runtime dependency mapping")
    require(
        runtime_dependencies.get("openpi_client")
        == {
            "accepted_package_initializer": (
                "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                "openpi-client/src/openpi_client/__init__.py"
            ),
            "accepted_package_initializer_sha256": OPENPI_CLIENT_INIT_SHA,
            "accepted_image_tools": (
                "$REPO_ROOT/.local/external/simulation/s4_3_pi1/openpi/packages/"
                "openpi-client/src/openpi_client/image_tools.py"
            ),
            "accepted_image_tools_sha256": OPENPI_CLIENT_IMAGE_TOOLS_SHA,
            "accepted_python_files": OPENPI_CLIENT_PYTHON_FILES,
            "accepted_python_tree_sha256": OPENPI_CLIENT_PYTHON_TREE_SHA,
            "required_distribution_version": "0.1.0",
            "runtime_origin_policy": ("REPOSITORY_SCOPED_AND_BYTE_IDENTICAL_TO_ACCEPTED_SOURCE"),
        },
        "OpenPI client runtime dependency contract",
    )
    require(
        runtime_dependencies.get("paligemma_tokenizer")
        == {
            "upstream_uri": "gs://big_vision/paligemma_tokenizer.model",
            "persistent_snapshot": (f"$PI2S_ROOT/{PALIGEMMA_TOKENIZER_SNAPSHOT}"),
            "persistent_manifest": (f"$PI2S_ROOT/{PALIGEMMA_TOKENIZER_MANIFEST}"),
            "persistent_manifest_sha256": PALIGEMMA_TOKENIZER_MANIFEST_SHA,
            "sha256": PALIGEMMA_TOKENIZER_SHA,
            "bytes": PALIGEMMA_TOKENIZER_BYTES,
            "required_sentencepiece_version": "0.2.2",
            "execution_cache_policy": (
                "COPY_VERIFIED_SNAPSHOT_TO_EPHEMERAL_OPENPI_DATA_HOME_NO_NETWORK"
            ),
        },
        "PaliGemma tokenizer runtime dependency contract",
    )
    require(
        runtime_dependencies.get("execution_environment")
        == {
            "pip_freeze_all_sha256": OPENPI_ENVIRONMENT_FREEZE_SHA,
            "identity_source": "historical starting_integrity environment freeze",
        },
        "OpenPI execution environment contract",
    )
    tokenizer_snapshot = pi2s_root / PALIGEMMA_TOKENIZER_SNAPSHOT
    require(
        tokenizer_snapshot.is_file()
        and not tokenizer_snapshot.is_symlink()
        and tokenizer_snapshot.stat().st_size == PALIGEMMA_TOKENIZER_BYTES
        and sha256_file(tokenizer_snapshot) == PALIGEMMA_TOKENIZER_SHA,
        "persistent PaliGemma tokenizer identity",
    )
    tokenizer_manifest = pi2s_root / PALIGEMMA_TOKENIZER_MANIFEST
    require(
        tokenizer_manifest.is_file()
        and not tokenizer_manifest.is_symlink()
        and sha256_file(tokenizer_manifest) == PALIGEMMA_TOKENIZER_MANIFEST_SHA,
        "persistent PaliGemma tokenizer manifest identity",
    )
    require(len(protocol["all_final_checkpoints"]) == 15, "fifteen finals")
    require(
        {(row["model"], int(row["training_seed"])) for row in protocol["all_final_checkpoints"]}
        == {
            (model, seed)
            for model in ("B0", "B_VA27", "B1", "B_HVA", "B2")
            for seed in (42, 43, 44)
        },
        "exact five-model three-seed final matrix",
    )
    require(
        all(
            row["directory_step"] == 29999
            and row["expected_restored_train_step"] == 30000
            and row["params_present"] is True
            and row["train_state_present"] is True
            and str(row["path"]).startswith("$EXPERIMENT_ROOT/")
            and is_sha256(row["tree_sha256"])
            for row in protocol["all_final_checkpoints"]
        ),
        "final checkpoint content and restore identities",
    )
    require(len(protocol["focus_checkpoints"]) == 6, "six focus checkpoints")
    final_by_key = {
        (row["model"], int(row["training_seed"])): row for row in protocol["all_final_checkpoints"]
    }
    require(
        tuple((row["model"], int(row["training_seed"])) for row in protocol["focus_checkpoints"])
        == FOCUS,
        "exact ordered focus checkpoint set",
    )
    require(
        all(
            row == final_by_key[(row["model"], int(row["training_seed"]))]
            for row in protocol["focus_checkpoints"]
        ),
        "focus records exactly match final-checkpoint authority",
    )
    require(len(protocol["intermediate_checkpoints"]) == 4, "four intermediate checkpoints")
    require(
        {
            (row["model"], int(row["training_seed"]), int(row["directory_step"]))
            for row in protocol["intermediate_checkpoints"]
        }
        == {
            ("B_HVA", 42, 10000),
            ("B_HVA", 42, 20000),
            ("B_HVA", 43, 10000),
            ("B_HVA", 43, 20000),
        },
        "exact HVA intermediate matrix",
    )
    require(
        all(
            row["closed_loop_authorized"] is False
            and row["identity_status"] == "FULL_CONTENT_HASH_VERIFIED"
            and row["identity_source_sha256"] == INTERMEDIATE_HASH_MANIFEST_SHA
            and str(row["path"]).startswith("$EXPERIMENT_ROOT/")
            and is_sha256(row["tree_sha256"])
            and int(row["files"]) > 0
            and int(row["bytes"]) > 0
            for row in protocol["intermediate_checkpoints"]
        ),
        "intermediate checkpoint content identities and no closed-loop authorization",
    )
    require(protocol["samples"]["train_windows_count"] == 512, "exact TRAIN budget")
    require(
        protocol["samples"]["policy_dev_windows_count"] == 256,
        "exact POLICY_DEV budget",
    )
    require(
        protocol["samples"]["train_source_groups_covered"] == 100, "TRAIN source-group coverage"
    )
    require(
        protocol["samples"]["policy_dev_source_groups_covered"] == 5,
        "POLICY_DEV source-group coverage",
    )
    samples = protocol["samples"]
    train_records = samples["train_window_records"]
    policy_records = samples["policy_dev_windows"]
    forbidden_outcome_keys = {
        "success",
        "failure",
        "reward",
        "termination",
        "terminated",
        "outcome",
    }
    require(
        len(train_records) == len(samples["train_windows"]) == 512
        and [int(row["row"]) for row in train_records] == samples["train_windows"],
        "TRAIN selected rows and records align",
    )
    require(
        len({row["source_group_id"] for row in train_records}) == 100
        and all(not (set(row) & forbidden_outcome_keys) for row in train_records),
        "TRAIN derived source groups and outcome-blind identities",
    )
    require(
        samples["train_window_records_sha256"] == canonical_sha(train_records)
        and samples["train_identity_keys_sha256"]
        == canonical_sha([canonical_sha(row) for row in train_records]),
        "TRAIN record and identity-key hashes",
    )
    require(
        len(policy_records) == 256
        and len({row["source_group_id"] for row in policy_records}) == 5
        and len({row["attempt_id"] for row in policy_records}) == 25
        and all(not (set(row) & forbidden_outcome_keys) for row in policy_records),
        "POLICY_DEV derived groups, attempts, and row-level outcome-blind identities",
    )
    require(
        samples["policy_dev_window_records_sha256"] == canonical_sha(policy_records)
        and samples["policy_dev_identity_keys_sha256"]
        == canonical_sha([row["selection_identity_sha256"] for row in policy_records]),
        "POLICY_DEV record and selection-key hashes",
    )
    require(
        len(protocol["samples"]["fixed_observation_rows"]) == 32, "exact fixed observation budget"
    )
    require(
        protocol["samples"]["fixed_observation_contact_strata"] == {"active": 16, "inactive": 16},
        "exact fixed observation strata",
    )
    require(
        protocol["samples"]["gradient_minibatches_per_checkpoint"] == 4
        and protocol["samples"]["gradient_batch_size"] == 2
        and len(protocol["samples"]["gradient_batches"]) == 4
        and all(len(batch) == 2 for batch in protocol["samples"]["gradient_batches"]),
        "exact gradient minibatch budget",
    )
    fixed_rows = {int(row) for row in samples["fixed_observation_rows"]}
    gradient_rows = [int(row) for batch in samples["gradient_batches"] for row in batch]
    require(fixed_rows <= set(samples["train_windows"]), "fixed rows are selected TRAIN rows")
    require(
        len(gradient_rows) == len(set(gradient_rows)) == 8 and set(gradient_rows) <= fixed_rows,
        "gradient rows are eight unique fixed TRAIN rows",
    )
    require(
        protocol["gradient_diagnostic"]["flow_times"] == [0.1, 0.5, 0.9],
        "exact gradient flow-time grid",
    )
    require(
        protocol["metrics"]["gradients"]["preprocess_mode"]
        == "DETERMINISTIC_EVAL_PREPROCESS_NO_AUGMENTATION",
        "deterministic gradient preprocessing mode",
    )
    require(
        protocol["development_rollouts"]["base"]["canonical_tuples"] == 72, "base rollout budget"
    )
    require(
        protocol["development_rollouts"]["optional_h_intervention"]["canonical_tuples"] == 48,
        "optional rollout budget",
    )
    require(protocol["development_rollouts"]["maximum_canonical_tuples"] == 120, "rollout maximum")
    require(protocol["training_budget"]["optimizer_updates"] == 0, "zero optimizer updates")
    require(
        protocol["samples"]["policy_dev_fixed_forward_compatibility"]
        == "NOT_COMPATIBLE_WITH_OFFICIAL_PI05_OBSERVATION_CONTRACT",
        "no fabricated POLICY_DEV official observations",
    )
    require(
        all(
            row["identity_status"] == "FULL_CONTENT_HASH_VERIFIED"
            for row in protocol["intermediate_checkpoints"]
        ),
        "intermediate checkpoint content identities",
    )
    require(protocol["gradient_diagnostic"]["optimizer_step"] is False, "no optimizer step")
    require(
        protocol["gradient_diagnostic"]["nuisance_noise_distribution"] == "jax.random.normal"
        and protocol["gradient_diagnostic"]["nuisance_noise_shape_per_minibatch"] == [2, 30, 32]
        and protocol["gradient_diagnostic"]["nuisance_noise_dtype"] == "float32",
        "exact gradient nuisance-noise contract",
    )
    require(
        protocol["prefix_integration_contract"]["adapter_output_shape_per_observation"] == [8, 2048]
        and protocol["prefix_integration_contract"]["input_mask_for_added_tokens"] is True
        and protocol["prefix_integration_contract"]["autoregressive_mask_for_added_tokens"]
        is False,
        "exact H prefix token and mask contract",
    )
    require(
        protocol["prefix_integration_contract"]["position_index_rule"] == "cumsum(input_mask)-1"
        and "padding does not advance"
        in protocol["prefix_integration_contract"]["padding_interaction"],
        "exact H prefix position and padding contract",
    )
    require(
        protocol["completion_policy"]["automatic_budget_expansion"] is False, "no budget expansion"
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.prepare_inputs and not args.write:
        parser.error("--prepare-inputs requires --write")
    if args.write:
        protocol = build_protocol(prepare_inputs=args.prepare_inputs)
        validate(protocol)
        atomic_json(args.output, protocol)
    else:
        protocol = json.loads(args.output.read_text())
    checks = validate(protocol)
    print(json.dumps({"status": "PASS", "checks": checks, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
