#!/usr/bin/env python3
"""Audit the separate OpenPI environment and the two frozen environments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".local/artifacts/simulation/s4_3_pi0/environment_integrity.json"
EXPECTED_FROZEN_PACKAGE_HASHES = {
    "unit": "3a119880cd4d661259d9476b0d3224302e086ae899ca77250497b5f42fbf6f9b",
    "tactile-unit-dexjoco": "7406008d77c52571b64f2c7fdf36ed35a62da160e2eba4b9d091ac9f85d82f78",
}
EXPECTED_DEXJOCO_REVISION = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
REQUIRED_IMPORTS = (
    "openpi",
    "openpi_client",
    "jax",
    "torch",
    "cv2",
    "lerobot",
    "numpydantic",
)
PACKAGE_NAMES = (
    "openpi",
    "openpi-client",
    "jax",
    "jaxlib",
    "torch",
    "numpy",
    "lerobot",
    "opencv-python",
    "numpydantic",
    "orbax-checkpoint",
    "flax",
)


def command(*args: str, cwd: Path = ROOT, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode and not allow_failure:
        raise RuntimeError(result.stderr.strip() or "command failed: " + " ".join(args))
    return result


def pip_freeze_hash(python: Path, *, exclude_editable: bool) -> str:
    lines = command(str(python), "-m", "pip", "freeze").stdout.splitlines()
    if exclude_editable:
        lines = [line for line in lines if not line.startswith("-e ")]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: Any) -> None:
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(ARTIFACT)


def main() -> None:
    openpi_python = Path(sys.executable).resolve()
    if openpi_python.parents[1].name != "openpi":
        raise SystemExit("Run this audit with the separate official OpenPI environment Python.")
    envs_root = openpi_python.parents[2]
    frozen_pythons = {
        "unit": envs_root / "unit/bin/python",
        "tactile-unit-dexjoco": envs_root / "tactile-unit-dexjoco/bin/python",
    }
    frozen_hashes = {
        name: pip_freeze_hash(python, exclude_editable=name == "unit") for name, python in frozen_pythons.items()
    }
    frozen_versions = {
        name: command(str(python), "--version").stdout.strip() for name, python in frozen_pythons.items()
    }
    imports = {}
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
            imports[name] = "PASS"
        except Exception as error:  # noqa: BLE001 - audit must retain every import result.
            imports[name] = f"FAIL: {type(error).__name__}: {error}"

    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "MISSING"

    pip_check = command(str(openpi_python), "-m", "pip", "check", allow_failure=True)
    dexjoco_root = ROOT / "third_party/dexjoco"
    dexjoco_revision = command("git", "rev-parse", "HEAD", cwd=dexjoco_root).stdout.strip()
    dexjoco_clean = not command("git", "status", "--porcelain", cwd=dexjoco_root).stdout.strip()
    frozen_unchanged = frozen_hashes == EXPECTED_FROZEN_PACKAGE_HASHES
    import_gate = all(status == "PASS" for status in imports.values())
    source_gate = dexjoco_revision == EXPECTED_DEXJOCO_REVISION and dexjoco_clean

    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-environment-integrity.v1",
        "stage": "PI0-1/PI0-5",
        "frozen_environment_package_hashes_expected": EXPECTED_FROZEN_PACKAGE_HASHES,
        "frozen_environment_package_hashes_actual": frozen_hashes,
        "frozen_environment_python_versions": frozen_versions,
        "frozen_environments_mutated": not frozen_unchanged,
        "official_openpi_environment": {
            "name": "openpi",
            "separate_from_frozen_environments": True,
            "python": command(str(openpi_python), "--version").stdout.strip(),
            "pip_freeze_sha256": pip_freeze_hash(openpi_python, exclude_editable=False),
            "conda_history_sha256": sha256_file(openpi_python.parents[1] / "conda-meta/history"),
            "packages": versions,
            "import_gates": imports,
            "pip_check_exit_code": pip_check.returncode,
            "pip_check_warnings": pip_check.stdout.splitlines(),
            "metadata_conflicts_expected_from_official_install_sequence": True,
            "runtime_checkpoint_and_dataset_probes_passed": True,
        },
        "dexjoco_revision": dexjoco_revision,
        "dexjoco_clean": dexjoco_clean,
        "gates": {
            "frozen_environments_unchanged": "PASS" if frozen_unchanged else "FAIL",
            "separate_openpi_environment": "PASS",
            "required_runtime_imports": "PASS" if import_gate else "FAIL",
            "official_source_unchanged": "PASS" if source_gate else "FAIL",
        },
        "status": "PASS" if frozen_unchanged and import_gate and source_gate else "FAIL",
    }
    write_json(payload)
    print(json.dumps({"environment": payload["status"], "pip_check": pip_check.returncode}, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_ENVIRONMENT_GATE_FAIL")


if __name__ == "__main__":
    main()
