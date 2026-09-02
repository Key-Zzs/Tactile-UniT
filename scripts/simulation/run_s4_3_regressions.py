#!/usr/bin/env python3
"""Run the final unit and DexJoCo S4.3 regression gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gr00t.simulation.s4_3_training import atomic_json, sha256_file  # noqa: E402

ARTIFACT_ROOT = ROOT / ".local/artifacts/simulation/s4_3_restart"
LOG_ROOT = ROOT / ".local/logs/simulation/s4_3_restart/regressions"
DEX_TESTS = (
    "tests/simulation/test_dexjoco_runtime.py",
    "tests/simulation/test_s4_3_pd_source_adapter.py",
    "tests/simulation/test_s4_3_pd_task_contracts.py",
    "tests/simulation/test_s4_3_pd_formal_protocol.py",
    "tests/simulation/test_s4_3_restart_protocol.py",
    "tests/simulation/test_s4_3_rollout_runtime.py",
)


def run(name: str, command: list[str], environment: dict[str, str]) -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"{name}.log"
    with path.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "log": str(path.relative_to(ROOT)),
        "log_sha256": sha256_file(path),
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def main() -> None:
    dex_python = os.environ.get("DEXJOCO_PYTHON")
    if not dex_python:
        raise RuntimeError("DEXJOCO_PYTHON is required")
    unit_environment = os.environ.copy()
    unit_environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    unit = run(
        "unit_pytest",
        [sys.executable, "-m", "pytest", "-q", "tests"],
        unit_environment,
    )
    dex_environment = os.environ.copy()
    dex_environment.pop("DISPLAY", None)
    dex_environment.update(
        {
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    dex = run(
        "dexjoco_applicable_pytest",
        [dex_python, "-m", "pytest", "-q", *DEX_TESTS],
        dex_environment,
    )
    dex_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT / "third_party/dexjoco",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dex_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT / "third_party/dexjoco",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    nested = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    diffusion = next(row for row in nested if "diffusion_policy" in row)
    result = {
        "schema": "tactile3d-unit.s4-3-regression-tests.v1",
        "stage": "R18",
        "unit": unit,
        "tactile_unit_dexjoco": dex,
        "robocasa_egl_smoke": {
            "evidence": dex["log"],
            "test": "test_real_pinch_tongs_headless_reset_step_rgb_contact_and_named_regions",
            "status": dex["status"],
        },
        "dexjoco_revision": dex_revision,
        "dexjoco_clean": not bool(dex_status.strip()),
        "nested_diffusion_policy": (
            "UNINITIALIZED" if diffusion.startswith("-") else "INITIALIZED_UNEXPECTEDLY"
        ),
        "failures": sum(item["status"] != "PASS" for item in (unit, dex)),
        "status": (
            "PASS"
            if unit["status"] == dex["status"] == "PASS"
            and not dex_status.strip()
            and diffusion.startswith("-")
            else "FAIL"
        ),
    }
    atomic_json(ARTIFACT_ROOT / "regression_tests.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit("S4_3_2 regression gate failed")


if __name__ == "__main__":
    main()
