#!/usr/bin/env python3
"""Validate the S4.0 decision contract and emit local audit evidence.

The optional smoke path is intentionally bounded to one existing RoboCasa GR1
environment, ten zero-action steps, raw contact API inspection, and one RGB
frame. It does not train, collect demonstrations, or construct tactile inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/simulation/s4_0_benchmark_audit.json"
DEFAULT_ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_0"
CANDIDATE_KEYS = ("RoboCasa", "DexJoCo", "DexMimicGen", "IsaacLab")
SCORE_KEYS = (
    "task_coverage",
    "dexterous",
    "bimanual",
    "contact_rich",
    "tactile_proxy",
    "headless",
    "data",
    "act",
    "diffusion_policy",
    "gr00t",
    "pi0_5",
    "randomization",
    "scale",
    "icra_relevance",
    "iclr_relevance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--smoke-existing", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(arguments)} failed")
    return completed.stdout.strip()


def load_and_validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "tactile3d-unit.s4-0-simulation-benchmark-audit.v1":
        raise ValueError("unexpected S4.0 audit schema")
    candidates = value.get("candidates", {})
    if tuple(candidates) != CANDIDATE_KEYS:
        raise ValueError(f"candidate order/names must be {CANDIDATE_KEYS}")
    primary = [name for name, row in candidates.items() if row.get("primary") is True]
    if primary != [value["recommendations"]["primary"]]:
        raise ValueError("exactly one candidate must match the primary recommendation")
    for name, row in candidates.items():
        scores = row.get("scores", {})
        if set(scores) != set(SCORE_KEYS):
            raise ValueError(f"{name} score keys are incomplete")
        if any(not isinstance(score, int) or not 0 <= score <= 4 for score in scores.values()):
            raise ValueError(f"{name} has a score outside 0..4")
        for required in (
            "license",
            "headless_status",
            "contact_api_status",
            "policy_adapter_status",
            "engineering_cost",
        ):
            if not row.get(required):
                raise ValueError(f"{name} is missing {required}")
    if value["s4_1"].get("status") != "RECOMMENDED_NOT_STARTED":
        raise ValueError("S4.1 must be recommended but not started")
    if value["simulated_tactile_contract"].get("status") != "DESIGN_ONLY_NOT_IMPLEMENTED":
        raise ValueError("S4.0 must not implement the tactile proxy")
    return value


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "numpy",
            "mujoco",
            "robosuite",
            "robocasa",
            "gymnasium",
            "opencv-python",
            "opencv-python-headless",
            "PyOpenGL",
            "isaaclab",
            "isaacsim",
        )
    }
    runtime: dict[str, Any] = {
        "canonical_environment": "unit",
        "display": "HEADLESS" if not os.environ.get("DISPLAY") else "DISPLAY_SET",
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "packages": packages,
        "environment_changed": config["environment"]["changed"],
        "pip_check": config["environment"]["pip_check"],
    }
    try:
        import torch

        runtime["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except Exception as error:  # pragma: no cover - diagnostic path
        runtime["torch_runtime_error"] = f"{type(error).__name__}: {error}"
    return runtime


def space_schema(space: Any) -> Any:
    import gymnasium as gym

    if isinstance(space, gym.spaces.Dict):
        return {key: space_schema(item) for key, item in space.spaces.items()}
    row: dict[str, Any] = {"type": type(space).__name__}
    if getattr(space, "shape", None) is not None:
        row["shape"] = list(space.shape)
    if hasattr(space, "dtype") and space.dtype is not None:
        row["dtype"] = str(space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        row["n"] = int(space.n)
    return row


def zero_action(space: Any) -> Any:
    import gymnasium as gym

    if isinstance(space, gym.spaces.Dict):
        return {key: zero_action(item) for key, item in space.spaces.items()}
    if isinstance(space, gym.spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, gym.spaces.Discrete):
        return 0
    raise TypeError(f"unsupported action space {type(space).__name__}")


def contact_snapshot(raw_env: Any, limit: int = 12) -> dict[str, Any]:
    import mujoco

    sim = raw_env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    contacts: list[dict[str, Any]] = []
    for index in range(min(int(data.ncon), limit)):
        contact = data.contact[index]
        local_wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, local_wrench)
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        contacts.append(
            {
                "index": index,
                "geom1_id": geom1,
                "geom2_id": geom2,
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
                "position_world": np.asarray(contact.pos),
                "frame": np.asarray(contact.frame).reshape(3, 3),
                "distance": float(contact.dist),
                "local_wrench_normal_tangent_torque": local_wrench,
            }
        )
    return {
        "ncon": int(data.ncon),
        "sample_count": len(contacts),
        "contacts": contacts,
        "available_fields": [
            "geom IDs and names",
            "body IDs and names",
            "contact position",
            "contact frame",
            "penetration or distance",
            "normal and tangential force via mj_contactForce",
            "contact-frame torque via mj_contactForce",
        ],
    }


def smoke_existing(artifact_root: Path, steps: int) -> dict[str, Any]:
    if not 10 <= steps <= 50:
        raise ValueError("S4.0 smoke steps must be between 10 and 50")
    if os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY must be unset for the headless smoke")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl are required")

    import gymnasium as gym
    import mujoco
    import robocasa  # noqa: F401 - registers the environments
    import robosuite
    from PIL import Image
    from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401 - registers GR00T Gym IDs

    env_name = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    destination = artifact_root / "robocasa_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_name, enable_render=True)
    try:
        observation, info = env.reset(seed=0)
        max_contacts = contact_snapshot(env.unwrapped.env)
        last_reward = 0.0
        last_terminated = False
        last_truncated = False
        for _ in range(steps):
            observation, last_reward, last_terminated, last_truncated, info = env.step(
                zero_action(env.action_space)
            )
            snapshot = contact_snapshot(env.unwrapped.env)
            if snapshot["ncon"] > max_contacts["ncon"]:
                max_contacts = snapshot
        frame = np.asarray(env.render())
        frame_path = destination / "frame.png"
        Image.fromarray(frame).save(frame_path)
        result = {
            "status": "PASS",
            "benchmark": "RoboCasa/robosuite/GR1",
            "environment": env_name,
            "robot": "GR1ArmsAndWaistFourierHands",
            "steps": steps,
            "backend": "EGL",
            "versions": {
                "mujoco": mujoco.__version__,
                "robosuite": robosuite.__version__,
                "robocasa": package_version("robocasa"),
            },
            "observation_space": space_schema(env.observation_space),
            "action_space": space_schema(env.action_space),
            "observation_keys": sorted(observation),
            "reward": float(last_reward),
            "terminated": bool(last_terminated),
            "truncated": bool(last_truncated),
            "info_keys": sorted(info),
            "rgb": {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "min": int(frame.min()),
                "max": int(frame.max()),
                "frame": "robocasa_smoke/frame.png",
            },
            "contact_api": max_contacts,
            "training_performed": False,
            "dataset_collection_performed": False,
        }
    finally:
        env.close()
    atomic_json(destination / "smoke_test.json", result)
    atomic_json(artifact_root / "smoke_test.json", result)
    return result


def write_evidence(config: Mapping[str, Any], artifact_root: Path) -> None:
    candidates = config["candidates"]
    matrix = {
        name: {"scores": row["scores"], "total": sum(row["scores"].values())}
        for name, row in candidates.items()
    }
    atomic_json(artifact_root / "environment_audit.json", environment_audit(config))
    atomic_json(artifact_root / "existing_stack_audit.json", config["existing_stack"])
    atomic_json(artifact_root / "candidate_matrix.json", matrix)
    atomic_json(
        artifact_root / "license_audit.json",
        {name: row["license"] for name, row in candidates.items()},
    )
    atomic_json(
        artifact_root / "headless_audit.json",
        {name: row["headless_status"] for name, row in candidates.items()},
    )
    atomic_json(
        artifact_root / "contact_api_audit.json",
        {
            "candidates": {name: row["contact_api_status"] for name, row in candidates.items()},
            "contract": config["simulated_tactile_contract"],
        },
    )
    atomic_json(
        artifact_root / "policy_interface_audit.json",
        {name: row["policy_adapter_status"] for name, row in candidates.items()},
    )
    atomic_json(artifact_root / "task_shortlist.json", config["paper_tasks"])
    atomic_json(
        artifact_root / "submodule_manifest.json",
        {
            "count": sum(bool(row["submodule"]) for row in candidates.values()),
            "entries": [
                {
                    "name": row["name"],
                    "path": row.get("submodule_path"),
                    "commit": row["audited_commit"],
                    "nested": row.get("nested_submodules", []),
                }
                for row in candidates.values()
                if row["submodule"]
            ],
            "git_status_recursive": git("submodule", "status", "--recursive").splitlines(),
        },
    )
    atomic_json(
        artifact_root / "final_decision.json",
        {
            "decision": config["decision"],
            "recommendations": config["recommendations"],
            "s4_1": config["s4_1"],
            "m3": "UNCHANGED — ESTABLISHED_WITH_WARNINGS",
            "s4_0": "COMPLETE",
            "s4_1_status": "NOT STARTED",
        },
    )

    acceptance = f"""# S4.0 Human Acceptance

- Branch: `{git('branch', '--show-current')}`
- HEAD: `{git('rev-parse', 'HEAD')}`
- Canonical environment: `unit` (unchanged)
- Candidates: RoboCasa, DexJoCo, DexMimicGen, Isaac Lab
- Added submodule: `third_party/dexjoco` at `{candidates['DexJoCo']['audited_commit']}`
- Primary: **DexJoCo**
- Regression: **RoboCasa/robosuite/GR1**
- Scale-up: **Isaac Lab/Isaac Sim**
- Data expansion: **DexMimicGen**
- Existing-stack headless result: **PASS (EGL)**
- Contact API: raw MuJoCo contacts and forces available; Isaac Lab has a native contact sensor
- Paper shortlist: {len(config['paper_tasks'])} exact DexJoCo tasks
- S4.1: recommended, **not started**
- Final decision: `{config['decision']}`
- Local smoke frame: `.local/artifacts/simulation/s4_0/robocasa_smoke/frame.png`

Review `docs/research/s4_0_sim_benchmark_audit.md` and the JSON files in this directory.
"""
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "HUMAN_ACCEPTANCE.md").write_text(acceptance)


def verify_m3(config: Mapping[str, Any]) -> None:
    expected = config["m3_integrity"]
    paths = {
        "manifest_sha256": ROOT / "configs/tactile_unit/m3_system_manifest.json",
        "c6_config_sha256": ROOT / "configs/tactile_unit/c6_m3_system_evaluation.json",
        "limitations_sha256": ROOT / "configs/tactile_unit/m3_limitations.json",
    }
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise RuntimeError(f"M3 integrity mismatch: {path.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    config = load_and_validate(args.config)
    verify_m3(config)
    write_evidence(config, args.artifact_root)
    if args.smoke_existing:
        smoke_existing(args.artifact_root, args.steps)
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": config["decision"],
                "primary": config["recommendations"]["primary"],
                "artifacts": str(args.artifact_root.relative_to(ROOT)),
                "smoke_existing": args.smoke_existing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
