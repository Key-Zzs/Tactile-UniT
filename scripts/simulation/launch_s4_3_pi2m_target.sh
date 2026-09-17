#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
unit_python="${UNIT_PYTHON:?Set UNIT_PYTHON to the frozen unit environment Python.}"
session="s43_pi2m_bhva_target_t27"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2m"
log_path="${repo_root}/.local/logs/simulation/s4_3_pi2m/bhva_target_t27.log"
output_path="${repo_root}/.local/datasets/simulation/s4_3_pi2m/pinch_tongs_va_t27/sidecar.npz"
launch_path="${artifact_dir}/target_build_launch.json"
status_path="${artifact_dir}/target_job_status.json"

if [[ -e "${log_path}" || -e "${output_path}" || -e "${launch_path}" || -e "${status_path}" ]]; then
  echo "Refusing to overwrite an existing PI2M target job/output." >&2
  exit 6
fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "Refusing to reuse existing tmux session ${session}." >&2
  exit 6
fi

snapshot1_gpu="$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
snapshot1_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
sleep 2
snapshot2_gpu="$(nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
snapshot2_apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"

declare -A busy_uuid=()
declare -A busy_uuid_first=()
while IFS=',' read -r uuid _pid _name _memory; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_uuid_first["${uuid}"]=1
done <<<"${snapshot1_apps}"
while IFS=',' read -r uuid _pid _name _memory; do
  uuid="${uuid// /}"
  [[ -n "${uuid}" ]] && busy_uuid["${uuid}"]=1
done <<<"${snapshot2_apps}"

selected=""
selected_uuid=""
declare -A idle_first=()
while IFS=',' read -r index uuid _name memory _total utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  [[ "${index}" =~ ^[0-3]$ ]] || continue
  if (( memory <= 64 && utilization <= 5 )) && [[ -z "${busy_uuid_first[${uuid}]:-}" ]]; then
    idle_first["${index}:${uuid}"]=1
  fi
done <<<"${snapshot1_gpu}"
while IFS=',' read -r index uuid _name memory _total utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  [[ "${index}" =~ ^[0-3]$ ]] || continue
  if (( memory <= 64 && utilization <= 5 )) \
    && [[ -z "${busy_uuid[${uuid}]:-}" ]] \
    && [[ -n "${idle_first[${index}:${uuid}]:-}" ]]; then
    lock_path="${common_git_dir}/tactile3d_unit_gpu${index}.lock"
    exec {probe_fd}>"${lock_path}"
    if flock -n "${probe_fd}"; then
      flock -u "${probe_fd}"
      selected="${index}"
      selected_uuid="${uuid}"
      break
    fi
  fi
done <<<"${snapshot2_gpu}"

if [[ -z "${selected}" ]]; then
  echo "No genuinely idle and unlockable GPU among physical GPU0-3." >&2
  exit 3
fi

mkdir -p "${artifact_dir}" "$(dirname "${log_path}")"
"${unit_python}" - "${launch_path}" "${selected}" "${selected_uuid}" "${session}" \
  "${log_path}" "${output_path}" "${snapshot1_gpu}" "${snapshot1_apps}" \
  "${snapshot2_gpu}" "${snapshot2_apps}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

path, gpu, uuid, session, log_path, output_path, gpu1, apps1, gpu2, apps2 = sys.argv[1:]
root = pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
payload = {
    "schema": "tactile3d-unit.s4-3-pi2m-target-build-launch.v1",
    "status": "LAUNCHING",
    "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "physical_gpu_id": int(gpu),
    "physical_gpu_uuid": uuid,
    "logical_cuda_device": 0,
    "session": session,
    "log_path": log_path,
    "output_path": output_path,
    "protocol_sha256": sha("configs/simulation/s4_3_pi2m_bhva_target_protocol.json"),
    "builder_sha256": sha("scripts/simulation/build_s4_3_pi2m_bhva_targets.py"),
    "auditor_sha256": sha("scripts/simulation/audit_s4_3_pi2m_bhva_targets.py"),
    "launcher_sha256": sha("scripts/simulation/launch_s4_3_pi2m_target.sh"),
    "supervisor_sha256": sha("scripts/simulation/supervise_s4_3_pi2m_target.sh"),
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

tmux new-session -d -s "${session}" \
  "cd '${repo_root}' && exec env UNIT_PYTHON='${unit_python}' PI2M_SESSION='${session}' bash scripts/simulation/supervise_s4_3_pi2m_target.sh '${selected}' '${selected_uuid}'"

for _ in $(seq 1 30); do
  [[ -e "${status_path}" ]] && break
  sleep 1
done
if [[ ! -e "${status_path}" ]]; then
  echo "PI2M target supervisor did not write status within 30 seconds." >&2
  exit 7
fi
cat "${status_path}"
