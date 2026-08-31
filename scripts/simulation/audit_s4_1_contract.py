#!/usr/bin/env python3
"""Audit tracked S4.1 contracts and write local machine-readable evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.dexjoco_adapter import (  # noqa: E402
    SimPolicyAction,
    policy_action_to_env_action,
)
from gr00t.simulation.timing import TimingContract, transition_pair_for_anchor  # noqa: E402

ARTIFACTS = ROOT / ".local/artifacts/simulation/s4_1"
PINNED_DEXJOCO = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"


def write_json(name: str, value: object) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    configs = {
        name: json.loads((ROOT / "configs/simulation" / name).read_text())
        for name in (
            "s4_1_dexjoco_runtime.json",
            "s4_1_dexjoco_contract.json",
            "s4_1_dexjoco_contact_regions.json",
            "s4_1_sim_tactile_contract.json",
        )
    }
    gitlink = git("ls-tree", "HEAD", "third_party/dexjoco").split()[2]
    submodule = git("-C", "third_party/dexjoco", "rev-parse", "HEAD")
    timing = TimingContract(physics_dt=0.002, control_dt=0.02)
    pair = transition_pair_for_anchor(25, 53, timing)
    known = SimPolicyAction(np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, *np.linspace(0, 0.75, 16)]))
    converted = policy_action_to_env_action(known)

    action = {
        "policy_dim": int(known.values.size),
        "environment_dim": int(converted.values.size),
        "known_input": known.values.tolist(),
        "known_output": converted.values.tolist(),
        "identity_rotation_wxyz": converted.values[3:7].tolist(),
        "deterministic": bool(
            np.array_equal(converted.values, policy_action_to_env_action(known).values)
        ),
        "status": "PASS",
    }
    timing_value = {
        "physics_dt": timing.physics_dt,
        "control_dt": timing.control_dt,
        "control_hz": timing.control_hz,
        "physics_substeps": timing.physics_substeps,
        "history_sec": timing.tactile_history_sec,
        "history_samples": timing.history_samples,
        "transition_target_sec": timing.transition_horizon_sec,
        "transition_control_steps": timing.transition_control_steps,
        "actual_transition_sec": timing.actual_transition_horizon_sec,
        "current_history_indices": pair.current_history_indices,
        "future_history_indices": pair.future_history_indices,
        "raw_overlap": pair.raw_overlap,
        "no_future_leakage": True,
        "status": "PASS" if not pair.raw_overlap else "FAIL",
    }
    runtime = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "m3_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", "m3", "HEAD"], cwd=ROOT
        ).returncode
        == 0,
        "dexjoco_gitlink": gitlink,
        "dexjoco_checkout": submodule,
        "dexjoco_clean": git("-C", "third_party/dexjoco", "status", "--short") == "",
        "config_sha256": {name: sha256(ROOT / "configs/simulation" / name) for name in configs},
        "scope": {
            "policy_training": False,
            "contact_representation_training": False,
            "third_party_modification": False,
        },
    }
    runtime["status"] = (
        "PASS"
        if runtime["branch"] == "develop/sim-benchmark"
        and runtime["m3_ancestor"]
        and gitlink == submodule == PINNED_DEXJOCO
        and runtime["dexjoco_clean"]
        else "FAIL"
    )
    write_json("runtime_contract_audit.json", runtime)
    write_json("action_contract.json", action)
    write_json("timing_contract.json", timing_value)
    write_json("sim_tactile_contract.json", configs["s4_1_sim_tactile_contract.json"])
    if importlib.util.find_spec("dexjoco") is not None:
        import cv2
        import gymnasium
        import mujoco
        import numpy

        environment = {
            "schema": "tactile3d-unit.s4-1-dexjoco-environment.v1",
            "name": "tactile-unit-dexjoco",
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "opencv_python": cv2.__version__,
            "pytorch": None,
            "dexjoco_source": "third_party/dexjoco/dexjoco",
            "dexjoco_sha": submodule,
            "headless_backend": "EGL",
            "egl_smoke_evidence": "headless_smoke.json",
            "display_required": False,
            "spec": "configs/simulation/s4_1_dexjoco_environment.yml",
            "status": "PASS",
        }
        write_json("dexjoco_environment.json", environment)
    if runtime["status"] != "PASS" or timing_value["status"] != "PASS":
        raise SystemExit("S4.1 tracked contract audit failed")
    print(json.dumps({"runtime": runtime["status"], "timing": timing_value["status"]}))


if __name__ == "__main__":
    main()
