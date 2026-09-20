#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
model_id="${1:?model ID required}"
gpu_id="${2:?physical GPU ID required}"
expected_uuid="${3:?physical GPU UUID required}"
run_root="${PI2N_RUN_ROOT:?Set PI2N_RUN_ROOT to the audited NAS run root.}"

case "${model_id}" in
  B_VA27) exp_name=s43_pi2n_b_va27_seed42 ;;
  B_VAC_V) exp_name=s43_pi2n_b_vac_v_seed42 ;;
  *) echo "Unsupported PI2N model ID: ${model_id}" >&2; exit 2 ;;
esac
model_key="$(tr '[:upper:]' '[:lower:]' <<<"${model_id}")"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2n"
log_dir="${repo_root}/.local/logs/simulation/s4_3_pi2n/${model_id}"
log_path="${log_dir}/train.log"
freeze_path="${artifact_dir}/${model_key}_training_freeze.json"
status_path="${run_root}/runtime_state/${model_id}/job_status.json"
heartbeat_path="${run_root}/runtime_state/${model_id}/heartbeat.json"
experiment_dir="${run_root}/runs/${model_id}/pinch_tongs/${exp_name}"

lambda_phys="$(${openpi_python} - "${freeze_path}" "${model_id}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "FROZEN_BEFORE_TRAINING"
assert payload["model_id"] == sys.argv[2]
assert payload["seed"] == 42 and payload["steps"] == 30000
assert payload["global_batch_size"] == 32
assert payload["checkpoint_selection"] == "FINAL_ONLY"
print(repr(payload["lambda_phys"]))
PY
)"

if [[ -e "${log_path}" || -e "${experiment_dir}" || -e "${status_path}" || -e "${heartbeat_path}" ]]; then
  echo "Refusing to overwrite existing PI2N run state for ${model_id}." >&2
  exit 6
fi
mkdir -p "${log_dir}" "$(dirname "${status_path}")"

write_json() {
  local path="$1"
  local state="$2"
  local message="$3"
  local exit_code="$4"
  "${openpi_python}" - "${path}" "${state}" "${message}" "${exit_code}" "${model_id}" "${gpu_id}" "${expected_uuid}" "${log_path}" "${experiment_dir}" "$$" <<'PY'
import datetime
import json
import pathlib
import sys

path, state, message, exit_code, model, gpu, uuid, log_path, experiment_dir, pid = sys.argv[1:]
payload = {
    "schema": "tactile3d-unit.s4-3-pi2n-runtime-state.v1",
    "state": state,
    "message": message,
    "exit_code": None if exit_code == "null" else int(exit_code),
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_id": model,
    "supervisor_pid": int(pid),
    "physical_gpu_id": int(gpu),
    "physical_gpu_uuid": uuid,
    "logical_cuda_device": 0,
    "log_path": log_path,
    "checkpoint_directory": experiment_dir,
    "expected_final_checkpoint": experiment_dir + "/29999",
}
destination = pathlib.Path(path)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
PY
}

lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
exec {lock_fd}>"${lock_path}"
if ! flock -n "${lock_fd}"; then
  write_json "${status_path}" FAILED "advisory GPU lock contended before training" 3
  exit 3
fi
actual_uuid="$(nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
utilization="$(nvidia-smi --id="${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
compute_apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
if [[ "${actual_uuid}" != "${expected_uuid}" ]] || (( memory_used > 64 || utilization > 5 )) || grep -Fxq "${expected_uuid}" <<<"${compute_apps}"; then
  write_json "${status_path}" FAILED "GPU identity or occupancy changed after advisory lock" 5
  exit 5
fi

write_json "${status_path}" RUNNING "GPU lock held; independent pi05_base seed42 30k run starting" null
write_json "${heartbeat_path}" STARTING "supervisor heartbeat active" null
(
  while kill -0 "$$" 2>/dev/null; do
    write_json "${heartbeat_path}" RUNNING "supervisor heartbeat active" null
    sleep 60
  done
) &
heartbeat_pid="$!"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
cd "${repo_root}"
set +e
"${openpi_python}" scripts/simulation/train_s4_3_pi2n.py \
  --model-id "${model_id}" \
  --exp-name "${exp_name}" \
  --lambda-phys "${lambda_phys}" \
  --fsdp-devices 1 \
  2>&1 | tee "${log_path}"
exit_code="${PIPESTATUS[0]}"
set -e
kill "${heartbeat_pid}" 2>/dev/null || true
wait "${heartbeat_pid}" 2>/dev/null || true
if (( exit_code == 0 )); then
  write_json "${status_path}" DONE "PI2N training completed" "${exit_code}"
  write_json "${heartbeat_path}" DONE "PI2N training completed" "${exit_code}"
else
  write_json "${status_path}" FAILED "PI2N training failed; inspect immutable log" "${exit_code}"
  write_json "${heartbeat_path}" FAILED "PI2N training failed; inspect immutable log" "${exit_code}"
fi
exit "${exit_code}"
