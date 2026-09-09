#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
physical_gpus="${PI1_PHYSICAL_GPUS:-1,2}"
frozen_config="${repo_root}/configs/simulation/s4_3_pi1c_frozen.json"
calibration="${repo_root}/.local/artifacts/simulation/s4_3_pi1/lambda_phys_calibration.json"
b1_freeze="${repo_root}/.local/artifacts/simulation/s4_3_pi1/pi1b_checkpoint_freeze.json"
session_log="${repo_root}/.local/logs/simulation/s4_3_pi1/pi1c/train.log"
experiment_dir="${repo_root}/.local/experiments/simulation/s4_3_pi1/training/pinch_tongs/s43_pi1c_contact_tokens_physical_aux_seed42"

lambda_phys="$("${openpi_python}" - "${frozen_config}" "${calibration}" "${b1_freeze}" <<'PY'
import hashlib
import json
import pathlib
import sys

config_path, calibration_path, freeze_path = map(pathlib.Path, sys.argv[1:])
config = json.loads(config_path.read_text())
calibration = json.loads(calibration_path.read_text())
freeze = json.loads(freeze_path.read_text())
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert config["status"] == "FROZEN_BEFORE_TRAINING"
assert config["mode"] == "CONTACT_STATE_TOKENS_PHYSICAL_AUX"
assert config["seed"] == 42 and config["steps"] == 30000 and config["global_batch_size"] == 32
assert calibration["status"] == "PASS" and calibration["frozen_before_PI1C_training"] is True
assert calibration["lambda_phys"] == config["lambda_phys"]
assert sha256(calibration_path) == config["calibration"]["artifact_sha256"]
assert freeze["status"] == "PASS" and freeze["frozen"] is True
assert sha256(freeze_path) == config["pi1b_checkpoint_freeze_sha256"]
print(repr(config["lambda_phys"]))
PY
)"

IFS=',' read -r -a gpu_ids <<<"${physical_gpus}"
if (( ${#gpu_ids[@]} == 0 || ${#gpu_ids[@]} > 3 || 32 % ${#gpu_ids[@]} != 0 )); then
  echo "The selected device count must be between 1 and 3 and divide global batch 32." >&2
  exit 2
fi

lock_fds=()
for gpu_id in "${gpu_ids[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU index: ${gpu_id}" >&2
    exit 2
  fi
  lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
  exec {lock_fd}>"${lock_path}"
  if ! flock -n "${lock_fd}"; then
    echo "GPU ${gpu_id} lock is already held: ${lock_path}" >&2
    exit 3
  fi
  lock_fds+=("${lock_fd}")
done

for gpu_id in "${gpu_ids[@]}"; do
  gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v wanted="${gpu_id}" '{gsub(/ /, "", $1); gsub(/^ +| +$/, "", $2); if ($1 == wanted) print $2}')"
  if [[ -z "${gpu_uuid}" ]]; then
    echo "Could not resolve GPU ${gpu_id}." >&2
    exit 4
  fi
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "${gpu_uuid}"; then
    echo "GPU ${gpu_id} became busy after lock acquisition." >&2
    exit 5
  fi
  memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( memory_used > 64 )); then
    echo "GPU ${gpu_id} has ${memory_used} MiB allocated after lock acquisition." >&2
    exit 5
  fi
done

if [[ -e "${session_log}" || -e "${experiment_dir}" ]]; then
  echo "Refusing to overwrite an existing PI1C log or experiment directory." >&2
  exit 6
fi

mkdir -p "$(dirname "${session_log}")" "$(dirname "${experiment_dir}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpus}"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

cd "${repo_root}"
"${openpi_python}" scripts/simulation/train_s4_3_pi1.py \
  --mode CONTACT_STATE_TOKENS_PHYSICAL_AUX \
  --exp-name s43_pi1c_contact_tokens_physical_aux_seed42 \
  --lambda-phys "${lambda_phys}" \
  2>&1 | tee "${session_log}"
