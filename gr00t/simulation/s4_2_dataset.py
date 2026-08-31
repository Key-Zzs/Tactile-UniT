"""Formal S4.2 DexJoCo dataset loading, pairing, and test-access guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / ".local/datasets/simulation/s4_2"
PRETEST_FREEZE = ROOT / ".local/artifacts/simulation/s4_2/pretest_freeze.json"


class FormalTestAccessError(RuntimeError):
    """Raised when formal test arrays are requested before the lock is frozen."""


@dataclass(frozen=True)
class EpisodeReference:
    episode_id: str
    task: str
    split: str
    source_trajectory_id: str
    directory: Path
    metadata: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_id(task: str, episode_id: str, anchor_step: int, future_step: int, split: str) -> str:
    return hashlib.sha256(
        f"{task}\0{episode_id}\0{anchor_step}\0{future_step}\0{split}".encode("utf-8")
    ).hexdigest()


def _validate_locked_test_access(pretest_freeze: Path) -> None:
    if not pretest_freeze.is_file():
        raise FormalTestAccessError(
            "formal test arrays are locked until pretest_freeze.json exists"
        )
    value = json.loads(pretest_freeze.read_text(encoding="utf-8"))
    required = {
        "test_loaded": False,
        "training_complete": True,
        "selection_complete": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise FormalTestAccessError(
            "pretest freeze does not authorize locked test evaluation"
        )


def episode_references(root: Path = DEFAULT_DATASET_ROOT) -> Iterator[EpisodeReference]:
    for metadata_path in sorted(Path(root).glob("episodes/*/metadata.json")):
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = value["metadata"]
        yield EpisodeReference(
            episode_id=value["episode_id"],
            task=metadata["task"],
            split=metadata["split"],
            source_trajectory_id=metadata["source_trajectory_id"],
            directory=metadata_path.parent,
            metadata=metadata,
        )


def references_for_split(
    split: str,
    root: Path = DEFAULT_DATASET_ROOT,
    *,
    purpose: str = "training_or_selection",
    pretest_freeze: Path = PRETEST_FREEZE,
) -> list[EpisodeReference]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and purpose != "dataset_quality":
        _validate_locked_test_access(pretest_freeze)
        if purpose not in {"locked_test", "deterministic_repeat"}:
            raise FormalTestAccessError(
                "formal test arrays may only be used by the locked evaluator"
            )
    return [reference for reference in episode_references(root) if reference.split == split]


def load_episode(reference: EpisodeReference) -> dict[str, np.ndarray]:
    with np.load(reference.directory / "steps.npz", allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def build_pair_arrays(
    split: str,
    root: Path = DEFAULT_DATASET_ROOT,
    *,
    purpose: str = "training_or_selection",
    pretest_freeze: Path = PRETEST_FREEZE,
) -> dict[str, np.ndarray]:
    references = references_for_split(
        split, root, purpose=purpose, pretest_freeze=pretest_freeze
    )
    rows: dict[str, list[np.ndarray | str | int]] = {
        "pair_id": [],
        "episode_id": [],
        "task": [],
        "source_trajectory_id": [],
        "anchor_step": [],
        "future_step": [],
        "current_history": [],
        "future_history": [],
        "teacher_future": [],
        "current_state": [],
        "action_chunk": [],
        "current_frame": [],
        "future_frame": [],
    }
    for reference in references:
        values = load_episode(reference)
        length = len(values["control_step"])
        for anchor in range(25, length - 27):
            future = anchor + 27
            rows["pair_id"].append(
                pair_id(reference.task, reference.episode_id, anchor, future, split)
            )
            rows["episode_id"].append(reference.episode_id)
            rows["task"].append(reference.task)
            rows["source_trajectory_id"].append(reference.source_trajectory_id)
            rows["anchor_step"].append(anchor)
            rows["future_step"].append(future)
            rows["current_history"].append(values["sim_tactile"][anchor - 25 : anchor + 1])
            rows["future_history"].append(values["sim_tactile"][anchor + 2 : future + 1])
            rows["teacher_future"].append(values["sim_tactile"][anchor + 1 : anchor + 14])
            rows["current_state"].append(values["proprio"][anchor, :23])
            rows["action_chunk"].append(values["policy_action"][anchor : future])
            rows["current_frame"].append(
                str(reference.directory / values["rgb_reference"][anchor])
            )
            rows["future_frame"].append(
                str(reference.directory / values["rgb_reference"][future])
            )
    return {
        name: np.asarray(value)
        for name, value in rows.items()
    }
