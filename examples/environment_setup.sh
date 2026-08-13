#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-unit}"
# Default: PyPI. Set PIP_INDEX_URL to a regional mirror if downloads are slow.
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if ! command -v conda >/dev/null 2>&1; then
  echo "error: conda not on PATH. Install Miniconda/Anaconda or source conda.sh first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

env_exists() {
  conda env list | awk -v n="$CONDA_ENV_NAME" '$1 == n { found=1 } END { exit !found }'
}

if env_exists; then
  echo "Conda env '$CONDA_ENV_NAME' already exists; skip conda create."
else
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"

pip install -i "$PIP_INDEX_URL" --upgrade setuptools
cd "$PROJECT_ROOT"
pip install -i "$PIP_INDEX_URL" einx
pip install -i "$PIP_INDEX_URL" -e ".[base]"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.1.post4}"
# PyPI provides only a source archive here. Shared GPU servers often have a
# working NVIDIA driver but no nvcc, so select the matching official wheel.
if [[ -n "${FLASH_ATTN_WHEEL_URL:-}" ]]; then
  pip install "$FLASH_ATTN_WHEEL_URL"
else
  FLASH_ATTN_WHEEL_URL="$(python - "$FLASH_ATTN_VERSION" <<'PY'
import platform
import sys
import torch

version = sys.argv[1]
if not sys.platform.startswith("linux") or platform.machine() not in {"x86_64", "amd64"}:
    raise SystemExit("Set FLASH_ATTN_WHEEL_URL for this platform")
if torch.version.cuda is None:
    raise SystemExit("The installed PyTorch build has no CUDA support")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cuda_major = torch.version.cuda.split(".")[0]
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
wheel = f"flash_attn-{version}+cu{cuda_major}torch{torch_version}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{wheel}")
PY
)"
  echo "Installing prebuilt FlashAttention wheel: $FLASH_ATTN_WHEEL_URL"
  if ! pip install "$FLASH_ATTN_WHEEL_URL"; then
    if command -v nvcc >/dev/null 2>&1; then
      pip install -i "$PIP_INDEX_URL" --no-build-isolation "flash-attn==$FLASH_ATTN_VERSION"
    else
      echo "error: no compatible wheel and nvcc is unavailable; set FLASH_ATTN_WHEEL_URL" >&2
      exit 1
    fi
  fi
fi
# pip install -i "$PIP_INDEX_URL" transformers==4.52.0
pip install -i "$PIP_INDEX_URL" "qwen-vl-utils[decord]==0.0.8"
pip install -i "$PIP_INDEX_URL" lpips

echo "Done. Activate with: conda activate $CONDA_ENV_NAME"
