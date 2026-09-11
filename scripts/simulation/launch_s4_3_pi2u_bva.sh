#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
frozen="${repo_root}/.local/artifacts/simulation/s4_3_pi2u/bva_training_protocol.json"
session_log="${repo_root}/.local/logs/simulation/s4_3_pi2u/bva/train.log"
experiment_dir="${repo_root}/.local/experiments/simulation/s4_3_pi2u/bva/pinch_tongs/s43_pi2u_bva_seed42"

lambda_phys="$("${openpi_python}" - "${frozen}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "FROZEN_BEFORE_TRAINING"
assert payload["intervention"] == "BVA"
assert payload["seed"] == 42 and payload["steps"] == 30000
assert payload["global_batch_size"] == 32
print(repr(payload["lambda_phys"]))
PY
)"

if [[ -e "${session_log}" || -e "${experiment_dir}" ]]; then
  echo "Refusing to overwrite an existing BVA log or experiment directory." >&2
  exit 6
fi

declare -A busy_uuid=()
while IFS=',' read -r uuid _pid _memory _name; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_uuid["${uuid}"]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true)

candidates=()
while IFS=',' read -r index uuid memory utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory// /}"
  utilization="${utilization// /}"
  if [[ "${index}" =~ ^[0-3]$ ]] && (( memory <= 64 && utilization <= 5 )) && [[ -z "${busy_uuid[${uuid}]:-}" ]]; then
    candidates+=("${index}")
  fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)

if (( ${#candidates[@]} == 0 )); then
  echo "No genuinely idle physical GPU is available." >&2
  exit 3
fi
desired=1
if (( ${#candidates[@]} >= 2 )); then desired=2; fi
selected=()
lock_fds=()
for gpu_id in "${candidates[@]}"; do
  lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
  exec {lock_fd}>"${lock_path}"
  if flock -n "${lock_fd}"; then
    selected+=("${gpu_id}")
    lock_fds+=("${lock_fd}")
  fi
  (( ${#selected[@]} == desired )) && break
done
if (( ${#selected[@]} == 0 )); then
  echo "Idle GPUs were found but all advisory locks were contended." >&2
  exit 3
fi
if (( ${#selected[@]} != desired )); then
  echo "Could not acquire the preregistered preferred device count after the scan." >&2
  exit 3
fi

for gpu_id in "${selected[@]}"; do
  gpu_uuid="$(nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "${gpu_uuid}"; then
    echo "GPU ${gpu_id} became busy after lock acquisition." >&2
    exit 5
  fi
  memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  utilization="$(nvidia-smi --id="${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if (( memory_used > 64 || utilization > 5 )); then
    echo "GPU ${gpu_id} ceased to be idle after lock acquisition." >&2
    exit 5
  fi
done

physical_gpus="$(IFS=,; echo "${selected[*]}")"
fsdp_devices="${#selected[@]}"
if (( fsdp_devices > 3 || 32 % fsdp_devices != 0 )); then
  echo "Selected device count violates the global-batch contract." >&2
  exit 2
fi

mkdir -p "$(dirname "${session_log}")" "$(dirname "${experiment_dir}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpus}"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

printf 'BVA_SELECTED_PHYSICAL_GPUS=%s FSDP_DEVICES=%s GLOBAL_BATCH=32\n' "${physical_gpus}" "${fsdp_devices}"
cd "${repo_root}"
"${openpi_python}" scripts/simulation/train_s4_3_pi2u_bva.py \
  --exp-name s43_pi2u_bva_seed42 \
  --lambda-phys "${lambda_phys}" \
  --fsdp-devices "${fsdp_devices}" \
  2>&1 | tee "${session_log}"
