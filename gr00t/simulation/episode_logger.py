"""Deterministic, conversion-friendly DexJoCo debug episode logger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dexjoco_adapter import SimEnvAction, SimObservation, SimPolicyAction


class DexJoCoEpisodeLogger:
    """Write one front-camera JPEG per step plus compressed numeric arrays."""

    def __init__(self, root: Path, episode_id: str, metadata: Mapping[str, Any]):
        self.root = Path(root)
        self.episode_id = episode_id
        self.episode_dir = self.root / episode_id
        self.frames_dir = self.episode_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.rows: list[dict[str, Any]] = []

    def append(
        self,
        observation: SimObservation,
        policy_action: SimPolicyAction,
        env_action: SimEnvAction,
        reward: float,
        diagnostics: Mapping[str, Any],
    ) -> None:
        import cv2

        if observation.episode_id != self.episode_id:
            raise ValueError("observation episode_id does not match logger")
        if self.rows and observation.timestamp_sec <= self.rows[-1]["timestamp_sec"]:
            raise ValueError("episode timestamps must be strictly increasing")
        reference = Path("frames") / f"{len(self.rows):06d}.jpg"
        destination = self.episode_dir / reference
        encoded = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(destination), encoded, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise OSError(f"failed to write RGB frame {destination}")
        self.rows.append(
            {
                "control_step": observation.control_step,
                "timestamp_sec": observation.timestamp_sec,
                "rgb_reference": reference.as_posix(),
                "proprio": observation.proprio.copy(),
                "policy_action": policy_action.values.copy(),
                "env_action": env_action.values.copy(),
                "sim_tactile": observation.sim_tactile.copy(),
                "reward": float(reward),
                "terminated": observation.terminated,
                "truncated": observation.truncated,
                "success": observation.success,
                "contact_count": int(diagnostics.get("contact_count", 0)),
                "contact_count_by_region": np.asarray(
                    list(diagnostics.get("contact_count_by_region", {}).values()),
                    dtype=np.int64,
                ),
            }
        )

    def finish(self) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("cannot finish an empty episode")
        arrays = {
            "episode_id": np.full(len(self.rows), self.episode_id),
            "task": np.full(len(self.rows), str(self.metadata.get("task", ""))),
            "seed": np.full(len(self.rows), int(self.metadata.get("seed", -1)), dtype=np.int64),
            "control_step": np.asarray([row["control_step"] for row in self.rows], dtype=np.int64),
            "timestamp_sec": np.asarray(
                [row["timestamp_sec"] for row in self.rows], dtype=np.float64
            ),
            "rgb_reference": np.asarray([row["rgb_reference"] for row in self.rows]),
            "proprio": np.stack([row["proprio"] for row in self.rows]),
            "policy_action": np.stack([row["policy_action"] for row in self.rows]),
            "env_action": np.stack([row["env_action"] for row in self.rows]),
            "sim_tactile": np.stack([row["sim_tactile"] for row in self.rows]),
            "reward": np.asarray([row["reward"] for row in self.rows], dtype=np.float32),
            "terminated": np.asarray([row["terminated"] for row in self.rows], dtype=bool),
            "truncated": np.asarray([row["truncated"] for row in self.rows], dtype=bool),
            "success": np.asarray(
                [False if row["success"] is None else row["success"] for row in self.rows],
                dtype=bool,
            ),
            "contact_count": np.asarray(
                [row["contact_count"] for row in self.rows], dtype=np.int64
            ),
            "contact_count_by_region": np.stack(
                [row["contact_count_by_region"] for row in self.rows]
            ),
        }
        np.savez_compressed(self.episode_dir / "steps.npz", **arrays)
        manifest = {
            "schema": "tactile3d-unit.dexjoco-debug-episode.v1",
            "episode_id": self.episode_id,
            "steps": len(self.rows),
            "storage": {"numeric": "steps.npz", "rgb": "frames/%06d.jpg"},
            "metadata": self.metadata,
            "fields": list(arrays),
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
