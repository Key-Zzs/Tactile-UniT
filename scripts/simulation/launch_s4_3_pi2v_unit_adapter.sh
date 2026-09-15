#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
unit_python="${UNIT_PYTHON:?Set UNIT_PYTHON to the audited UniT environment Python.}"
protocol="${repo_root}/configs/simulation/s4_3_pi2v_unit_adapter_protocol.json"
metrics_log="${repo_root}/.local/logs/simulation/s4_3_pi2v/unit_adapter/train.jsonl"
stdout_log="${repo_root}/.local/logs/simulation/s4_3_pi2v/unit_adapter/train.stdout.log"
experiment="${repo_root}/.local/experiments/simulation/s4_3_pi2v/unit_adapter"

[[ "$(git branch --show-current)" == "develop/sim-benchmark" ]] || {
  echo "PI2V formal launch requires develop/sim-benchmark." >&2; exit 2;
}
[[ -z "$(git status --short)" ]] || {
  echo "PI2V formal launch requires a clean committed worktree." >&2; exit 2;
}

"${unit_python}" - "${protocol}" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert protocol["status"] == "FROZEN_BEFORE_FORMAL_TRAINING"
assert protocol["seed"] == 42
assert protocol["optimization"]["world_size"] == 2
assert protocol["optimization"]["effective_global_batch"] == 256
assert protocol["optimization"]["optimizer_steps"] == 80_000
artifact_root = pathlib.Path(sys.argv[1]).parents[2] / ".local/artifacts/simulation/s4_3_pi2v"
for name in (
    "starting_integrity.json",
    "historical_immutability_before.json",
    "official_unit_revalidation.json",
    "temporal_contract_audit.json",
    "adapter_normalization.json",
    "adapter_data_contract.json",
):
    assert json.loads((artifact_root / name).read_text())["status"] == "PASS"
PY

if [[ -e "${metrics_log}" || -e "${stdout_log}" || -e "${experiment}" ]]; then
  echo "Refusing to overwrite a formal PI2V adapter log or experiment." >&2
  exit 6
fi

declare -A busy_uuid=()
while IFS=',' read -r uuid _pid _memory _name; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_uuid["${uuid}"]=1
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits 2>/dev/null || true)

candidates=()
while IFS=',' read -r index uuid total memory utilization; do
  index="${index// /}"; uuid="${uuid// /}"; total="${total// /}"
  memory="${memory// /}"; utilization="${utilization// /}"
  if (( total >= 48000 && memory <= 64 && utilization <= 5 )) && [[ -z "${busy_uuid[${uuid}]:-}" ]]; then
    candidates+=("${index}")
  fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits)

if (( ${#candidates[@]} < 2 )); then
  echo "Two genuinely idle >=48 GiB GPUs are required by the frozen protocol." >&2
  exit 3
fi

selected=()
lock_fds=()
for gpu_id in "${candidates[@]}"; do
  lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
  exec {lock_fd}>"${lock_path}"
  if flock -n "${lock_fd}"; then
    selected+=("${gpu_id}")
    lock_fds+=("${lock_fd}")
  fi
  (( ${#selected[@]} == 2 )) && break
done
if (( ${#selected[@]} != 2 )); then
  echo "Idle GPUs were found but two advisory locks could not be acquired." >&2
  exit 4
fi

for gpu_id in "${selected[@]}"; do
  gpu_uuid="$(nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "${gpu_uuid}"; then
    echo "GPU ${gpu_id} became busy after lock acquisition." >&2
    exit 5
  fi
  read -r memory utilization < <(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  if (( memory > 64 || utilization > 5 )); then
    echo "GPU ${gpu_id} ceased to be idle after lock acquisition." >&2
    exit 5
  fi
done

physical_gpus="$(IFS=,; echo "${selected[*]}")"
mkdir -p "$(dirname "${metrics_log}")" "$(dirname "${experiment}")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${physical_gpus}"
export PYTHONUNBUFFERED=1
printf 'PI2V_SELECTED_PHYSICAL_GPUS=%s WORLD_SIZE=2 EFFECTIVE_BATCH=256\n' "${physical_gpus}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

# Locks protect selection and the mandatory post-lock idle recheck. Release
# them before the long process so no advisory lock is held at the user pause;
# the CUDA contexts themselves make device occupancy visible to later scans.
for lock_fd in "${lock_fds[@]}"; do flock -u "${lock_fd}"; done

cd "${repo_root}"
"${unit_python}" -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/simulation/train_s4_3_pi2v_unit_adapter.py \
  --mode formal \
  --output "${experiment}" \
  --metrics-log "${metrics_log}" \
  --max-steps 80000 \
  --microbatch 32 \
  --gradient-accumulation 4 \
  --effective-batch 256 \
  --save-steps 20000 \
  --seed 42 \
  --startup-audit-step 55 \
  2>&1 | tee "${stdout_log}"
