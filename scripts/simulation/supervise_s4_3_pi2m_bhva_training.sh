#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
gpu_csv="${1:?comma-separated physical GPU IDs required}"
uuid_csv="${2:?comma-separated physical GPU UUIDs required}"
session="${PI2M_SESSION:-s43_pi2m_bhva_s42}"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2m"
log_dir="${repo_root}/.local/logs/simulation/s4_3_pi2m/bhva"
log_path="${log_dir}/train.log"
status_path="${artifact_dir}/job_status.json"
done_path="${artifact_dir}/TRAINING_DONE"
failed_path="${artifact_dir}/TRAINING_FAILED"
freeze_path="${artifact_dir}/training_protocol_freeze.json"

IFS=',' read -r -a gpu_ids <<<"${gpu_csv}"
IFS=',' read -r -a expected_uuids <<<"${uuid_csv}"
if (( ${#gpu_ids[@]} != ${#expected_uuids[@]} )) || (( ${#gpu_ids[@]} == 0 )); then
  echo "GPU ID/UUID cardinality mismatch." >&2
  exit 2
fi

lambda_phys="$(${openpi_python} - "${freeze_path}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "FROZEN_BEFORE_TRAINING"
assert payload["model_id"] == "B_HVA"
assert payload["seed"] == 42 and payload["steps"] == 30000
assert payload["global_batch_size"] == 32
print(repr(payload["lambda_phys"]))
PY
)"

mkdir -p "${artifact_dir}" "${log_dir}"

write_status() {
  local state="$1"
  local exit_code="$2"
  local message="$3"
  "${openpi_python}" - "${status_path}" "${state}" "${exit_code}" "${message}" \
    "${gpu_csv}" "${uuid_csv}" "${session}" "${log_path}" "$$" <<'PY'
import datetime
import json
import pathlib
import sys

path, state, exit_code, message, gpus, uuids, session, log_path, pid = sys.argv[1:]
payload = {
    "schema": "tactile3d-unit.s4-3-pi2m-training-job-status.v1",
    "state": state,
    "exit_code": None if exit_code == "null" else int(exit_code),
    "message": message,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "supervisor_pid": int(pid),
    "session": session,
    "physical_gpu_ids": [int(value) for value in gpus.split(",")],
    "physical_gpu_uuids": uuids.split(","),
    "logical_cuda_devices": list(range(len(gpus.split(",")))),
    "log_path": log_path,
    "checkpoint_directory": "$REPO_ROOT/.local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42",
    "expected_final_checkpoint": "$REPO_ROOT/.local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42/29999",
}
destination = pathlib.Path(path)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
PY
}

lock_fds=()
for gpu_id in "${gpu_ids[@]}"; do
  lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
  exec {lock_fd}>"${lock_path}"
  if ! flock -n "${lock_fd}"; then
    write_status FAILED 3 "advisory GPU lock contended before training"
    touch "${failed_path}"
    exit 3
  fi
  lock_fds+=("${lock_fd}")
done

compute_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
for offset in "${!gpu_ids[@]}"; do
  gpu_id="${gpu_ids[$offset]}"
  expected_uuid="${expected_uuids[$offset]}"
  actual_uuid="$(nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
  memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  utilization="$(nvidia-smi --id="${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${actual_uuid}" != "${expected_uuid}" ]] || (( memory_used > 64 || utilization > 5 )) \
    || grep -Eq "^${expected_uuid}," <<<"${compute_apps}"; then
    write_status FAILED 5 "GPU identity or occupancy changed after advisory locks"
    touch "${failed_path}"
    exit 5
  fi
done

rm -f "${done_path}" "${failed_path}"
write_status RUNNING null "GPU locks held; unique B_HVA seed42 30k run starting"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_csv}"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
cd "${repo_root}"
set +e
"${openpi_python}" scripts/simulation/train_s4_3_pi2m_bhva.py \
  --exp-name s43_pi2m_bhva_seed42 \
  --lambda-phys "${lambda_phys}" \
  --fsdp-devices "${#gpu_ids[@]}" \
  2>&1 | tee "${log_path}"
exit_code="${PIPESTATUS[0]}"
set -e
if (( exit_code == 0 )); then
  write_status DONE "${exit_code}" "unique B_HVA seed42 training completed"
  touch "${done_path}"
else
  write_status FAILED "${exit_code}" "B_HVA training failed; inspect frozen log"
  touch "${failed_path}"
fi
exit "${exit_code}"
