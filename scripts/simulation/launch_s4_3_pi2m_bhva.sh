#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
session="s43_pi2m_bhva_s42"
artifact_dir="${repo_root}/.local/artifacts/simulation/s4_3_pi2m"
log_path="${repo_root}/.local/logs/simulation/s4_3_pi2m/bhva/train.log"
experiment_dir="${repo_root}/.local/experiments/simulation/s4_3_pi2m/bhva/pinch_tongs/s43_pi2m_bhva_seed42"
launch_path="${artifact_dir}/training_launch.json"
status_path="${artifact_dir}/job_status.json"
freeze_path="${artifact_dir}/training_protocol_freeze.json"

if [[ ! -f "${freeze_path}" ]]; then
  echo "Frozen PI2M training protocol is missing." >&2
  exit 4
fi
if [[ -e "${log_path}" || -e "${experiment_dir}" || -e "${launch_path}" || -e "${status_path}" ]]; then
  echo "Refusing to overwrite an existing PI2M training run." >&2
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

candidates=()
candidate_uuids=()
while IFS=',' read -r index uuid _name memory _total utilization; do
  index="${index// /}"
  uuid="${uuid// /}"
  memory="${memory//[^0-9]/}"
  utilization="${utilization//[^0-9]/}"
  [[ "${index}" =~ ^[0-3]$ ]] || continue
  if (( memory <= 64 && utilization <= 5 )) \
    && [[ -z "${busy_second[${uuid}]:-}" ]] \
    && [[ -n "${idle_first[${index}:${uuid}]:-}" ]]; then
    candidates+=("${index}")
    candidate_uuids+=("${uuid}")
  fi
done <<<"${snapshot2_gpu}"

if (( ${#candidates[@]} >= 4 )); then
  desired=4
elif (( ${#candidates[@]} >= 2 )); then
  desired=2
elif (( ${#candidates[@]} == 1 )); then
  desired=1
else
  echo "No genuinely idle GPU among physical GPU0-3." >&2
  exit 3
fi

selected=()
selected_uuids=()
for offset in "${!candidates[@]}"; do
  gpu_id="${candidates[$offset]}"
  lock_path="${common_git_dir}/tactile3d_unit_gpu${gpu_id}.lock"
  exec {probe_fd}>"${lock_path}"
  if flock -n "${probe_fd}"; then
    flock -u "${probe_fd}"
    selected+=("${gpu_id}")
    selected_uuids+=("${candidate_uuids[$offset]}")
  fi
  (( ${#selected[@]} == desired )) && break
done
if (( ${#selected[@]} != desired )); then
  echo "Could not acquire the preferred 1/2/4-device advisory-lock set." >&2
  exit 3
fi

gpu_csv="$(IFS=,; echo "${selected[*]}")"
uuid_csv="$(IFS=,; echo "${selected_uuids[*]}")"
mkdir -p "${artifact_dir}" "$(dirname "${log_path}")"
"${openpi_python}" - "${launch_path}" "${gpu_csv}" "${uuid_csv}" "${session}" \
  "${log_path}" "${experiment_dir}" "${snapshot1_gpu}" "${snapshot1_apps}" \
  "${snapshot2_gpu}" "${snapshot2_apps}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

path, gpus, uuids, session, log_path, experiment_dir, gpu1, apps1, gpu2, apps2 = sys.argv[1:]
root = pathlib.Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
sha = lambda relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
freeze = json.loads((root / ".local/artifacts/simulation/s4_3_pi2m/training_protocol_freeze.json").read_text())
assert freeze["status"] == "FROZEN_BEFORE_TRAINING"
payload = {
    "schema": "tactile3d-unit.s4-3-pi2m-training-launch.v1",
    "status": "LAUNCHING",
    "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "physical_gpu_ids": [int(value) for value in gpus.split(",")],
    "physical_gpu_uuids": uuids.split(","),
    "logical_cuda_devices": list(range(len(gpus.split(",")))),
    "fsdp_devices": len(gpus.split(",")),
    "global_batch_size": 32,
    "session": session,
    "log_path": log_path,
    "experiment_directory": experiment_dir,
    "expected_final_checkpoint": experiment_dir + "/29999",
    "lambda_phys": freeze["lambda_phys"],
    "training_protocol_sha256": hashlib.sha256(
        (root / ".local/artifacts/simulation/s4_3_pi2m/training_protocol_freeze.json").read_bytes()
    ).hexdigest(),
    "code_sha256": {
        relative: sha(relative)
        for relative in (
            "gr00t/simulation/pi05_tactile_unit.py",
            "gr00t/simulation/s4_3_pi1.py",
            "scripts/simulation/train_s4_3_pi2m_bhva.py",
            "scripts/simulation/launch_s4_3_pi2m_bhva.sh",
            "scripts/simulation/supervise_s4_3_pi2m_bhva_training.sh",
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

tmux new-session -d -s "${session}" \
  "cd '${repo_root}' && exec env OPENPI_PYTHON='${openpi_python}' PI2M_SESSION='${session}' bash scripts/simulation/supervise_s4_3_pi2m_bhva_training.sh '${gpu_csv}' '${uuid_csv}'"

for _ in $(seq 1 30); do
  [[ -e "${status_path}" ]] && break
  sleep 1
done
if [[ ! -e "${status_path}" ]]; then
  echo "PI2M training supervisor did not write status within 30 seconds." >&2
  exit 7
fi
cat "${status_path}"
