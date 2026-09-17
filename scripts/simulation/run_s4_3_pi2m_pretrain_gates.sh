#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2m"
log_dir="${repo_root}/.local/logs/simulation/s4_3_pi2m"
launch_path="${artifact_dir}/pretrain_gate_launch.json"
gradient_log="${log_dir}/loaded_base_gradient_gate.log"
calibration_log="${log_dir}/lambda_calibration.log"

for output in \
  "${launch_path}" \
  "${artifact_dir}/loaded_base_gradient_gate.json" \
  "${artifact_dir}/lambda_calibration.json" \
  "${gradient_log}" \
  "${calibration_log}"; do
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite PI2M pretraining gate output: ${output}" >&2
    exit 6
  fi
done

snapshot1_gpu="$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
snapshot1_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
sleep 2
snapshot2_gpu="$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
snapshot2_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"

declare -A busy_first=()
declare -A busy_second=()
while IFS=',' read -r uuid _pid _name _memory; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_first["${uuid}"]=1
done <<<"${snapshot1_apps}"
while IFS=',' read -r uuid _pid _name _memory; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_second["${uuid}"]=1
done <<<"${snapshot2_apps}"

declare -A idle_first=()
while IFS=',' read -r index uuid _name memory _total utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  [[ "${index}" =~ ^[0-3]$ ]] || continue
  if (( memory <= 64 && utilization <= 5 )) && [[ -z "${busy_first[${uuid}]:-}" ]]; then
    idle_first["${index}:${uuid}"]=1
  fi
done <<<"${snapshot1_gpu}"

selected=""
selected_uuid=""
lock_fd=""
while IFS=',' read -r index uuid _name memory _total utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  [[ "${index}" =~ ^[0-3]$ ]] || continue
  if (( memory <= 64 && utilization <= 5 )) \
    && [[ -z "${busy_second[${uuid}]:-}" ]] \
    && [[ -n "${idle_first[${index}:${uuid}]:-}" ]]; then
    lock_path="${common_git_dir}/tactile3d_unit_gpu${index}.lock"
    exec {candidate_fd}>"${lock_path}"
    if flock -n "${candidate_fd}"; then
      selected="${index}"
      selected_uuid="${uuid}"
      lock_fd="${candidate_fd}"
      break
    fi
  fi
done <<<"${snapshot2_gpu}"

if [[ -z "${selected}" ]]; then
  echo "No genuinely idle and unlockable GPU among physical GPU0-3." >&2
  exit 3
fi

actual_uuid="$(nvidia-smi --id="${selected}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')"
memory_used="$(nvidia-smi --id="${selected}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
utilization="$(nvidia-smi --id="${selected}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
compute_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ "${actual_uuid}" != "${selected_uuid}" ]] || (( memory_used > 64 || utilization > 5 )) \
  || grep -Eq "^${selected_uuid}," <<<"${compute_apps}"; then
  echo "Selected GPU ceased to be idle after lock acquisition." >&2
  exit 5
fi

mkdir -p "${artifact_dir}" "${log_dir}"
"${openpi_python}" - "${launch_path}" "${selected}" "${selected_uuid}" \
  "${snapshot1_gpu}" "${snapshot1_apps}" "${snapshot2_gpu}" "${snapshot2_apps}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

path, gpu, uuid, gpu1, apps1, gpu2, apps2 = sys.argv[1:]
root = pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
sha = lambda relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
payload = {
    "schema": "tactile3d-unit.s4-3-pi2m-pretrain-gate-launch.v1",
    "status": "RUNNING",
    "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "physical_gpu_id": int(gpu),
    "physical_gpu_uuid": uuid,
    "logical_cuda_device": 0,
    "code_sha256": {
        relative: sha(relative)
        for relative in (
            "gr00t/simulation/pi05_tactile_unit.py",
            "gr00t/simulation/s4_3_pi1.py",
            "scripts/simulation/train_s4_3_pi2m_bhva.py",
            "scripts/simulation/validate_s4_3_pi2m_loaded_base_gradients.py",
            "scripts/simulation/calibrate_s4_3_pi2m_bhva_lambda.py",
            "configs/simulation/s4_3_pi2m_bhva_protocol.json",
        )
    },
    "snapshots": [
        {"gpu": gpu1.splitlines(), "compute_apps": apps1.splitlines()},
        {"gpu": gpu2.splitlines(), "compute_apps": apps2.splitlines()},
    ],
}
destination = pathlib.Path(path)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(destination)
PY

mark_status() {
  local state="$1"
  local exit_code="$2"
  "${openpi_python}" - "${launch_path}" "${state}" "${exit_code}" <<'PY'
import datetime
import json
import pathlib
import sys

path, state, exit_code = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
payload = json.loads(path.read_text())
payload["status"] = state
payload["exit_code"] = exit_code
payload["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}
trap 'exit_code=$?; if (( exit_code != 0 )); then mark_status FAILED "${exit_code}"; fi' EXIT

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${selected}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
cd "${repo_root}"
echo "PI2M_PRETRAIN_GATE_PHYSICAL_GPU=${selected} UUID=${selected_uuid} LOGICAL_GPU=0"
"${openpi_python}" scripts/simulation/validate_s4_3_pi2m_loaded_base_gradients.py \
  2>&1 | tee "${gradient_log}"
"${openpi_python}" scripts/simulation/calibrate_s4_3_pi2m_bhva_lambda.py \
  2>&1 | tee "${calibration_log}"

mark_status DONE 0
trap - EXIT
