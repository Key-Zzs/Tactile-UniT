#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
unit_python="${UNIT_PYTHON:?Set UNIT_PYTHON to the frozen unit environment Python.}"
gpu_id="${1:?physical GPU id required}"
expected_uuid="${2:?physical GPU UUID required}"
session="${PI2M_SESSION:-s43_pi2m_bhva_target_t27}"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2m"
log_dir="${repo_root}/.local/logs/simulation/s4_3_pi2m"
log_path="${log_dir}/bhva_target_t27.log"
status_path="${artifact_dir}/target_job_status.json"
done_path="${artifact_dir}/TARGET_BUILD_DONE"
failed_path="${artifact_dir}/TARGET_BUILD_FAILED"
lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"

mkdir -p "${artifact_dir}" "${log_dir}"

write_status() {
  local state="$1"
  local exit_code="$2"
  local message="$3"
  "${unit_python}" - "${status_path}" "${state}" "${exit_code}" "${message}" \
    "${gpu_id}" "${expected_uuid}" "${session}" "${log_path}" "$$" <<'PY'
import datetime
import json
import pathlib
import sys

path, state, exit_code, message, gpu, uuid, session, log_path, pid = sys.argv[1:]
payload = {
    "schema": "tactile3d-unit.s4-3-pi2m-target-job-status.v1",
    "state": state,
    "exit_code": None if exit_code == "null" else int(exit_code),
    "message": message,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "supervisor_pid": int(pid),
    "session": session,
    "physical_gpu_id": int(gpu),
    "physical_gpu_uuid": uuid,
    "logical_cuda_device": 0,
    "log_path": log_path,
    "output_path": "$REPO_ROOT/.local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz",
}
destination = pathlib.Path(path)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
PY
}

exec {lock_fd}>"${lock_path}"
if ! flock -n "${lock_fd}"; then
  write_status "FAILED" 3 "advisory GPU lock contended before target build"
  touch "${failed_path}"
  exit 3
fi

actual_uuid="$(nvidia-smi --id="${gpu_id}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
memory_used="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
utilization="$(nvidia-smi --id="${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
compute_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ "${actual_uuid}" != "${expected_uuid}" ]] || (( memory_used > 64 || utilization > 5 )) \
  || grep -Eq "^${expected_uuid}," <<<"${compute_apps}"; then
  write_status "FAILED" 5 "GPU identity or occupancy changed after advisory lock acquisition"
  touch "${failed_path}"
  exit 5
fi

rm -f "${done_path}" "${failed_path}"
write_status "RUNNING" null "GPU lock held; corrected +27 clean VA target build starting"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "${repo_root}"
set +e
"${unit_python}" scripts/simulation/build_s4_3_pi2m_bhva_targets.py 2>&1 | tee "${log_path}"
exit_code="${PIPESTATUS[0]}"
set -e
if (( exit_code == 0 )); then
  write_status "DONE" "${exit_code}" "corrected +27 clean VA target build completed"
  touch "${done_path}"
else
  write_status "FAILED" "${exit_code}" "corrected +27 clean VA target build failed; inspect log"
  touch "${failed_path}"
fi
exit "${exit_code}"
