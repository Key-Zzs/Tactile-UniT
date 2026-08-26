#!/usr/bin/env bash
# Hold a shared per-GPU lock and expose exactly one allowed A-R device.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 64
fi

common_git_dir="$(
  git rev-parse --path-format=absolute --git-common-dir 2>/dev/null ||
    (cd "$(git rev-parse --git-common-dir)" && pwd)
)"

nvidia-smi -i 2
nvidia-smi -i 3
nvidia-smi \
  --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader

try_gpu() {
  local physical_gpu="$1"
  shift
  local lock_path="${common_git_dir}/tactile3d_unit_gpu${physical_gpu}.lock"
  (
    if ! flock -n 9; then
      exit 73
    fi
    if [[ -n "$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
      exit 74
    fi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${physical_gpu}"
    export TACTILE_PHYSICAL_GPU="${physical_gpu}"
    export TACTILE_GPU_LOCK_NAME="tactile3d_unit_gpu${physical_gpu}.lock"
    exec "$@"
  ) 9>"${lock_path}"
}

for physical_gpu in 3 2; do
  set +e
  try_gpu "${physical_gpu}" "$@"
  status=$?
  set -e
  if [[ ${status} -eq 73 || ${status} -eq 74 ]]; then
    continue
  fi
  exit "${status}"
done

echo "GPU_RESOURCE_BUSY: physical GPUs 3 and 2 are locked or have compute processes" >&2
exit 75
