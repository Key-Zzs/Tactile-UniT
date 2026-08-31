#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
  echo "conda is not available; set CONDA_EXE to the existing installation" >&2
  exit 2
fi

env_name="tactile-unit-dexjoco"
spec="configs/simulation/s4_1_dexjoco_environment.yml"
if "$conda_exe" env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  "$conda_exe" run -n "$env_name" python -c '
from pathlib import Path
import dexjoco
import gymnasium
import mujoco
import numpy

root = Path.cwd().resolve()
source = Path(dexjoco.__file__).resolve()
expected = (root / "third_party/dexjoco/dexjoco").resolve()
if expected not in source.parents:
    raise SystemExit(f"existing environment has unrelated DexJoCo source: {source}")
expected_versions = {
    "mujoco": (mujoco.__version__, "3.4.0"),
    "numpy": (numpy.__version__, "1.26.4"),
    "gymnasium": (gymnasium.__version__, "1.0.0"),
}
wrong = {name: values for name, values in expected_versions.items() if values[0] != values[1]}
if wrong:
    raise SystemExit(f"existing environment has incompatible versions: {wrong}")
print("existing tactile-unit-dexjoco environment provenance: PASS")
'
else
  "$conda_exe" env create -f "$spec"
fi
