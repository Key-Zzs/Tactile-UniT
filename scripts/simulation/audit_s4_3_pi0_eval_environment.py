#!/usr/bin/env python3
"""Record the isolated official DexJoCo/OpenPI client environment."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_PYTHON = ROOT / ".local/external/s4_3_pi0/eval-venv/bin/python"
DEXJOCO = ROOT / "third_party/dexjoco"
OUTPUT = ROOT / ".local/artifacts/simulation/s4_3_pi0/evaluation_environment.json"


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    probe = r"""
import importlib
import importlib.metadata
import json
packages = {
    "dexjoco": "dexjoco",
    "openpi-client": "openpi_client",
    "dm-robotics-transformations": "dm_robotics",
    "mujoco": "mujoco",
    "numpy": "numpy",
    "scipy": "scipy",
    "websockets": "websockets",
    "msgpack": "msgpack",
    "imageio": "imageio",
    "pyyaml": "yaml",
    "tyro": "tyro",
}
result = {}
for distribution, module_name in packages.items():
    module = importlib.import_module(module_name)
    result[distribution] = {
        "version": importlib.metadata.version(distribution),
        "module_file": getattr(module, "__file__", None),
    }
print(json.dumps(result, sort_keys=True))
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(DEXJOCO / "dexjoco"),
    }
    result = subprocess.run(
        [str(EVAL_PYTHON), "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    packages = json.loads(result.stdout)
    dexjoco_revision = command("git", "rev-parse", "HEAD", cwd=DEXJOCO)
    dexjoco_clean = not command("git", "status", "--porcelain", cwd=DEXJOCO)
    source_gates = {
        "dexjoco": str(DEXJOCO / "dexjoco") in packages["dexjoco"]["module_file"],
        "openpi-client": str(DEXJOCO / "openpi/packages/openpi-client/src")
        in packages["openpi-client"]["module_file"],
    }
    versions = {name: value["version"] for name, value in packages.items()}
    gates = {
        "isolated_overlay_exists": EVAL_PYTHON.is_file(),
        "official_dexjoco_source": source_gates["dexjoco"],
        "official_openpi_client_source": source_gates["openpi-client"],
        "official_openpi_client_version": versions["openpi-client"] == "0.1.0",
        "numpy_1_26_4": versions["numpy"] == "1.26.4",
        "mujoco_3_4_0": versions["mujoco"] == "3.4.0",
        "dexjoco_revision_frozen": dexjoco_revision == "8d23b0fab23b17a58c4b55f3942e17013aaf8267",
        "dexjoco_clean": dexjoco_clean,
    }
    payload = {
        "schema": "tactile3d-unit.s4-3-pi0-evaluation-environment.v1",
        "stage": "PI0-10",
        "python": "$EVAL_ENV/bin/python",
        "base_environment": "tactile-unit-dexjoco (read-only system site packages)",
        "overlay": "$REPO_ROOT/.local/external/s4_3_pi0/eval-venv",
        "installation": (
            "official openpi-client editable source installed into an isolated overlay; "
            "no package was installed into either frozen environment"
        ),
        "package_versions": versions,
        "source_gates": {name: "PASS" if passed else "FAIL" for name, passed in source_gates.items()},
        "dexjoco_revision": dexjoco_revision,
        "gates": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit("S4_3_PI0_EVALUATION_ENVIRONMENT_FAIL")


if __name__ == "__main__":
    main()
