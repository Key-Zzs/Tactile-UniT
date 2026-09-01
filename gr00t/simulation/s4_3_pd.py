"""Official demonstration adaptation and logging for S4.3-PD."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dexjoco_adapter import (
    SimEnvAction,
    SimObservation,
    SimPolicyAction,
    policy_action_to_env_action,
    proprio_to_neutral_policy_action,
)

TASKS = ("pinch_tongs", "hammer_nail", "click_mouse")
REQUIRED_SOURCE_FIELDS = ("action", "action_rotvec", "state", "timestamp")
PILOT_GROUPS_PER_TASK = 4
FORMAL_GROUPS_PER_TASK = 25
ATTEMPTS_PER_GROUP = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def quaternion_wxyz_to_signed_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Invert a quaternion without discarding its sign at the policy boundary.

    DexJoCo's controller is sensitive to quaternion sign continuity even though
    q and -q encode the same mathematical rotation. Returning angles in [0, 2π]
    lets the existing central rotvec→quaternion adapter reproduce the recorded
    official quaternion rather than silently canonicalizing its sign.
    """

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("source quaternion must be finite wxyz with shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("source quaternion norm is zero")
    quaternion = quaternion / norm
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-12:
        if quaternion[0] < 0:
            return np.asarray([2.0 * np.pi, 0.0, 0.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, float(quaternion[0]))
    return quaternion[1:] * (angle / vector_norm)


def official_env_action_to_policy_action(action: np.ndarray) -> SimPolicyAction:
    """Adapt official [xyz, quat-wxyz, hand16] through the central contract."""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (23,) or not np.isfinite(action).all():
        raise ValueError("official single-arm source action must be finite 23D")
    rotvec = quaternion_wxyz_to_signed_rotvec(action[3:7])
    policy = SimPolicyAction(np.concatenate([action[:3], rotvec, action[7:23]]))
    reconstructed = policy_action_to_env_action(policy).values.astype(np.float64)
    source_quaternion = action[3:7] / np.linalg.norm(action[3:7])
    if not np.allclose(reconstructed[:3], action[:3], atol=2e-6):
        raise ValueError("central adapter failed official TCP xyz round-trip")
    if not np.allclose(reconstructed[7:], action[7:], atol=2e-6):
        raise ValueError("central adapter failed official hand Action round-trip")
    if float(np.dot(reconstructed[3:7], source_quaternion)) < 1.0 - 2e-6:
        raise ValueError("central adapter failed signed quaternion round-trip")
    return policy


def policy_proprio_22(proprio: np.ndarray) -> np.ndarray:
    """Filter privileged task state and express robot proprio in 22D form."""

    values = proprio_to_neutral_policy_action(np.asarray(proprio, dtype=np.float64)).values
    if values.shape != (22,) or not np.isfinite(values).all():
        raise ValueError("policy proprio conversion did not produce finite 22D state")
    return values


@dataclass(frozen=True)
class OfficialReplaySource:
    task: str
    source_group_id: str
    replay_path: Path
    source_tree_sha256: str
    environment_actions: np.ndarray
    policy_actions: tuple[SimPolicyAction, ...]
    initial_state: np.ndarray
    source_timestamps: np.ndarray

    @property
    def steps(self) -> int:
        return len(self.policy_actions)

    @classmethod
    def load(cls, task: str, replay_path: Path) -> "OfficialReplaySource":
        if task not in TASKS:
            raise ValueError(f"unsupported task {task!r}")
        import zarr

        replay_path = Path(replay_path)
        data = zarr.open(str(replay_path), mode="r")["data"]
        missing = sorted(set(REQUIRED_SOURCE_FIELDS) - set(data.array_keys()))
        if missing:
            raise ValueError(f"source replay is missing fields: {missing}")
        environment_actions = np.asarray(data["action"], dtype=np.float64)
        initial_state = np.asarray(data["state"][0], dtype=np.float64).ravel()
        timestamps = np.asarray(data["timestamp"], dtype=np.float64).ravel()
        if environment_actions.ndim != 2 or environment_actions.shape[1] != 23:
            raise ValueError(f"source action shape is {environment_actions.shape}, not [T,23]")
        if initial_state.shape[0] < 23:
            raise ValueError("source state does not contain 23D robot proprio")
        if timestamps.shape != (len(environment_actions),):
            raise ValueError("source timestamps do not align with Actions")
        if not all(
            np.isfinite(value).all()
            for value in (environment_actions, initial_state, timestamps)
        ):
            raise ValueError("source replay contains NaN or Inf")
        policy_actions = tuple(
            official_env_action_to_policy_action(action) for action in environment_actions
        )
        return cls(
            task=task,
            source_group_id=replay_path.parent.name,
            replay_path=replay_path,
            source_tree_sha256=replay_tree_sha256(replay_path),
            environment_actions=environment_actions,
            policy_actions=policy_actions,
            initial_state=initial_state,
            source_timestamps=timestamps,
        )


def discover_official_sources(task: str, root: Path) -> list[Path]:
    task_root = Path(root) / task
    candidates = sorted(task_root.glob("*/replay.zarr"))
    complete = [
        path
        for path in candidates
        if all((path / "data" / field).exists() for field in REQUIRED_SOURCE_FIELDS)
    ]
    required = PILOT_GROUPS_PER_TASK + FORMAL_GROUPS_PER_TASK
    if len(complete) < required:
        raise RuntimeError(f"{task} has {len(complete)} complete sources; {required} required")
    return complete[:required]


def source_partition(task: str, root: Path, role: str) -> list[Path]:
    sources = discover_official_sources(task, root)
    if role == "PD_PILOT":
        return sources[:PILOT_GROUPS_PER_TASK]
    if role == "FORMAL":
        return sources[PILOT_GROUPS_PER_TASK:]
    raise ValueError(f"unknown source role {role!r}")


def expert_action_at(
    source: OfficialReplaySource, step: int
) -> tuple[SimPolicyAction, str, int]:
    if step < source.steps:
        return source.policy_actions[step], "REPLAY", step
    if source.task == "pinch_tongs":
        # The native task counts three open→close cycles.  Some official
        # demonstrations cross the third open threshold by only a few
        # hundredths of a radian in float64 and can miss it after the frozen
        # policy-facing float32 round trip.  Reuse the source's late open hand
        # pose, while retaining its final lifted TCP target, before returning
        # to the source's final close pose.  This remains an Action-only
        # physical replay adaptation: no object or task state is mutated.
        recovery_step = step - source.steps
        if recovery_step < 30:
            open_index = int(round(0.88 * (source.steps - 1)))
            values = source.policy_actions[-1].values.copy()
            values[6:] = source.policy_actions[open_index].values[6:]
            return SimPolicyAction(values), "RECOVERY_OPEN", open_index
        if recovery_step < 60:
            return source.policy_actions[-1], "RECOVERY_CLOSE", source.steps - 1
    return source.policy_actions[-1], "HOLD_SUCCESS", source.steps - 1


def native_task_state(
    task: str, raw_env: Any, info: Mapping[str, Any]
) -> tuple[np.ndarray, tuple[str, ...], float]:
    if task == "pinch_tongs":
        tongs_z = float(raw_env._data.sensor("tongs_pos").data[2])
        values = np.asarray(
            [
                raw_env._pinch_count,
                tongs_z,
                raw_env._lift_z,
                getattr(raw_env, "_success_counter", 0),
            ],
            dtype=np.float64,
        )
        progress = min(1.0, float(raw_env._pinch_count) / 3.0)
        if tongs_z < raw_env._lift_z:
            progress *= 0.75
        names = ("pinch_count", "tongs_z", "lift_z", "success_counter")
    elif task == "hammer_nail":
        face_z = (
            float(raw_env._data.geom_xpos[raw_env._face_gid][2])
            if raw_env._face_gid >= 0
            else np.nan
        )
        values = np.asarray(
            [raw_env._nail_depth, raw_env._success_depth, bool(info.get("hammer_hit")), face_z],
            dtype=np.float64,
        )
        progress = min(1.0, float(raw_env._nail_depth / raw_env._success_depth))
        names = ("nail_depth", "success_depth", "hammer_hit", "hammer_face_z")
    else:
        try:
            joint_delta = float(raw_env._data.sensor("mouse_joint0_pos").data) - float(
                raw_env._mouse_joint0_init
            )
        except Exception:
            joint_delta = np.nan
        values = np.asarray(
            [
                raw_env._mouse_in_mousepad(),
                raw_env._display_blue,
                raw_env._success_trigger_count,
                joint_delta,
            ],
            dtype=np.float64,
        )
        progress = min(
            1.0, float(raw_env._success_trigger_count / raw_env._success_trigger_target)
        )
        if raw_env._display_blue:
            progress = max(progress, 0.5)
        names = ("mouse_in_mousepad", "display_blue", "success_counter", "button_delta")
    return values, names, float(progress)


class S43PDAttemptLogger:
    """Store student-visible transitions plus acquisition-only diagnostics."""

    def __init__(self, attempt_dir: Path, metadata: Mapping[str, Any], *, jpeg_quality: int = 80):
        self.attempt_dir = Path(attempt_dir)
        self.frames_dir = self.attempt_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.jpeg_quality = int(jpeg_quality)
        self.rows: list[dict[str, Any]] = []

    def append(
        self,
        current: SimObservation,
        action: SimPolicyAction,
        env_action: SimEnvAction,
        next_observation: SimObservation,
        reward: float,
        info: Mapping[str, Any],
        phase: str,
        source_step: int,
        current_contact_diagnostics: Mapping[str, Any],
        native_state: np.ndarray,
        task_progress: float,
    ) -> None:
        import cv2

        if self.rows and current.timestamp_sec <= self.rows[-1]["timestamp_sec"]:
            raise ValueError("attempt timestamps must be strictly increasing")
        frame_reference = Path("frames") / f"{len(self.rows):06d}.jpg"
        destination = self.attempt_dir / frame_reference
        bgr = cv2.cvtColor(current.rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(
            str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ):
            raise OSError(f"failed to write frame {destination}")
        tactile = np.asarray(current.sim_tactile, dtype=np.float32)
        if tactile.shape != (30,):
            raise ValueError(f"student tactile must be 30D, got {tactile.shape}")
        normal_force = float(np.sum(tactile[1::6]))
        tangential_force = float(np.sum(tactile[2::6]))
        self.rows.append(
            {
                "control_step": current.control_step,
                "timestamp_sec": current.timestamp_sec,
                "next_timestamp_sec": next_observation.timestamp_sec,
                "rgb_reference": frame_reference.as_posix(),
                "proprio": policy_proprio_22(current.proprio),
                "sim_tactile": tactile,
                "policy_action": action.values.copy(),
                "env_action": env_action.values.copy(),
                "reward": float(reward),
                "terminated": next_observation.terminated,
                "truncated": next_observation.truncated,
                "success": bool(next_observation.success),
                "expert_phase": phase,
                "source_step": int(source_step),
                "contact_count": int(current_contact_diagnostics.get("contact_count", 0)),
                "normal_force": normal_force,
                "tangential_force": tangential_force,
                "native_task_state": np.asarray(native_state, dtype=np.float64),
                "task_progress": float(task_progress),
            }
        )

    def finish(
        self, *, termination_reason: str, native_state_names: Sequence[str]
    ) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("cannot finish empty acquisition attempt")
        arrays = {
            "control_step": np.asarray([row["control_step"] for row in self.rows], dtype=np.int64),
            "timestamp_sec": np.asarray(
                [row["timestamp_sec"] for row in self.rows], dtype=np.float64
            ),
            "next_timestamp_sec": np.asarray(
                [row["next_timestamp_sec"] for row in self.rows], dtype=np.float64
            ),
            "rgb_reference": np.asarray([row["rgb_reference"] for row in self.rows]),
            "proprio": np.stack([row["proprio"] for row in self.rows]),
            "sim_tactile": np.stack([row["sim_tactile"] for row in self.rows]),
            "policy_action": np.stack([row["policy_action"] for row in self.rows]),
            "env_action": np.stack([row["env_action"] for row in self.rows]),
            "reward": np.asarray([row["reward"] for row in self.rows], dtype=np.float32),
            "terminated": np.asarray([row["terminated"] for row in self.rows], dtype=bool),
            "truncated": np.asarray([row["truncated"] for row in self.rows], dtype=bool),
            "success": np.asarray([row["success"] for row in self.rows], dtype=bool),
            "expert_phase": np.asarray([row["expert_phase"] for row in self.rows]),
            "source_step": np.asarray([row["source_step"] for row in self.rows], dtype=np.int64),
            "contact_count": np.asarray(
                [row["contact_count"] for row in self.rows], dtype=np.int64
            ),
            "normal_force": np.asarray(
                [row["normal_force"] for row in self.rows], dtype=np.float32
            ),
            "tangential_force": np.asarray(
                [row["tangential_force"] for row in self.rows], dtype=np.float32
            ),
            "native_task_state": np.stack([row["native_task_state"] for row in self.rows]),
            "task_progress": np.asarray(
                [row["task_progress"] for row in self.rows], dtype=np.float32
            ),
        }
        for name, array in arrays.items():
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"attempt field {name} contains NaN/Inf")
        numeric_path = self.attempt_dir / "steps.npz"
        np.savez_compressed(numeric_path, **arrays)
        rgb_digest = hashlib.sha256()
        for frame in sorted(self.frames_dir.glob("*.jpg")):
            rgb_digest.update(frame.name.encode("utf-8"))
            rgb_digest.update(bytes.fromhex(sha256_file(frame)))
        success = bool(np.any(arrays["success"]))
        metadata = dict(self.metadata)
        metadata.update(
            {
                "termination_reason": termination_reason,
                "native_task_state_names": list(native_state_names),
                "student_observation_fields": {
                    "rgb": "rgb_reference",
                    "proprio": [22],
                    "tactile": [30],
                    "expert_action": [22],
                },
                "privileged_state_is_diagnostic_only": True,
            }
        )
        manifest = {
            "schema": "tactile3d-unit.s4-3-pd-attempt.v1",
            "attempt_id": str(metadata["attempt_id"]),
            "steps": len(self.rows),
            "success": success,
            "metadata": metadata,
            "fields": {name: list(array.shape) for name, array in arrays.items()},
            "checksums": {
                "steps_npz_sha256": sha256_file(numeric_path),
                "rgb_tree_sha256": rgb_digest.hexdigest(),
            },
        }
        (self.attempt_dir / "metadata.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
