"""Repository-owned DexJoCo runtime adapter for S4.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .simulated_tactile import ContactRegionMap, SimulatedTactileExtractor
from .timing import TimingContract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_CONFIG = ROOT / "configs/simulation/s4_1_dexjoco_contact_regions.json"


@dataclass(frozen=True)
class SimPolicyAction:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (22,):
            raise ValueError(f"single-arm policy action must be 22D, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("single-arm policy action must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SimEnvAction:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (23,):
            raise ValueError(f"single-arm environment action must be 23D, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("single-arm environment action must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SimObservation:
    timestamp_sec: float
    control_step: int
    episode_id: str
    task_name: str
    rgb: np.ndarray
    proprio: np.ndarray
    sim_tactile: np.ndarray
    terminated: bool
    truncated: bool
    success: bool | None = None


def policy_action_to_env_action(action: SimPolicyAction | np.ndarray) -> SimEnvAction:
    """Convert [xyz, rotvec, hand16] to [xyz, quaternion-wxyz, hand16]."""

    values = (
        action.values if isinstance(action, SimPolicyAction) else SimPolicyAction(action).values
    )
    quat_xyzw = Rotation.from_rotvec(values[3:6].astype(np.float64)).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    converted = np.concatenate([values[:3], quat_wxyz, values[6:22]])
    return SimEnvAction(converted)


def proprio_to_neutral_policy_action(proprio: np.ndarray) -> SimPolicyAction:
    """Construct the official hold/stay action from the current 23D robot state."""

    state = np.asarray(proprio, dtype=np.float64)
    if state.shape[0] < 23:
        raise ValueError("DexJoCo single-arm proprio must contain at least 23 values")
    quat_wxyz = state[3:7]
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rotvec = Rotation.from_quat(quat_xyzw).as_rotvec()
    return SimPolicyAction(np.concatenate([state[:3], rotvec, state[7:23]]))


class DexJoCoRuntimeAdapter:
    """Own lifecycle, contracts, contact extraction, action conversion, and rendering."""

    def __init__(
        self,
        task_name: str = "pinch_tongs",
        seed: int = 0,
        episode_id: str = "episode-000",
        camera_name: str = "front",
        randomize: bool = False,
        region_config: Path = DEFAULT_REGION_CONFIG,
    ):
        self.task_name = task_name
        self.seed = int(seed)
        self.episode_id = episode_id
        self.camera_name = camera_name
        self.randomize = bool(randomize)
        self.region_config = Path(region_config)
        self.env: Any = None
        self._mujoco: Any = None
        self.control_step = 0
        self.last_raw_observation: dict[str, np.ndarray] | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self.region_map: ContactRegionMap | None = None
        self.tactile_extractor: SimulatedTactileExtractor | None = None
        self.timing: TimingContract | None = None

    @property
    def raw_env(self) -> Any:
        if self.env is None:
            raise RuntimeError("adapter is not started")
        return self.env.unwrapped

    def start(self) -> dict[str, Any]:
        import mujoco
        from dexjoco.tasks import CONFIG_MAPPING

        if self.task_name not in CONFIG_MAPPING:
            raise ValueError(f"unknown DexJoCo task {self.task_name!r}")
        config = CONFIG_MAPPING[self.task_name]()
        self.env = config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=self.randomize,
            randomize_dynamics=False,
            seed=self.seed,
        )
        self._mujoco = mujoco
        value = json.loads(self.region_config.read_text())
        self.region_map = ContactRegionMap.from_config(value)
        region_audit = self.region_map.resolve(self.raw_env.model, mujoco)
        self.tactile_extractor = SimulatedTactileExtractor(self.region_map)
        self.timing = TimingContract(
            physics_dt=float(self.raw_env.physics_dt), control_dt=float(self.raw_env.control_dt)
        )
        return region_audit

    def reset(self) -> SimObservation:
        if self.env is None:
            raise RuntimeError("call start() before reset()")
        raw_observation, info = self.env.reset()
        self.control_step = 0
        self.last_raw_observation = raw_observation
        return self._observation(raw_observation, False, False, info.get("succeed"))

    def neutral_policy_action(self) -> SimPolicyAction:
        if self.last_raw_observation is None:
            raise RuntimeError("reset the adapter before requesting a neutral action")
        return proprio_to_neutral_policy_action(self.last_raw_observation["state"])

    def step(
        self, action: SimPolicyAction | np.ndarray
    ) -> tuple[SimObservation, float, dict[str, Any], SimEnvAction]:
        if self.env is None:
            raise RuntimeError("call start() before step()")
        policy_action = action if isinstance(action, SimPolicyAction) else SimPolicyAction(action)
        env_action = policy_action_to_env_action(policy_action)
        raw_observation, reward, terminated, truncated, info = self.env.step(env_action.values)
        self.control_step += 1
        self.last_raw_observation = raw_observation
        observation = self._observation(
            raw_observation, bool(terminated), bool(truncated), info.get("succeed")
        )
        return observation, float(reward), dict(info), env_action

    def _observation(
        self,
        raw_observation: dict[str, np.ndarray],
        terminated: bool,
        truncated: bool,
        success: Any,
    ) -> SimObservation:
        assert self.tactile_extractor is not None
        tactile, diagnostics = self.tactile_extractor.extract(
            self.raw_env.model, self.raw_env.data, self._mujoco
        )
        self.last_diagnostics = diagnostics
        return SimObservation(
            timestamp_sec=float(self.raw_env.data.time),
            control_step=self.control_step,
            episode_id=self.episode_id,
            task_name=self.task_name,
            rgb=np.asarray(raw_observation[self.camera_name], dtype=np.uint8),
            proprio=np.asarray(raw_observation["state"], dtype=np.float64),
            sim_tactile=tactile,
            terminated=terminated,
            truncated=truncated,
            success=None if success is None else bool(success),
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
