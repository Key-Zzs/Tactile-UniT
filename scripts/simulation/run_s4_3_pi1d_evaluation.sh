#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
common_git_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
gpu_index="${PI1_EVAL_GPU:-1}"
openpi_python="${OPENPI_PYTHON:?Set OPENPI_PYTHON to the frozen openpi environment Python.}"
unit_python="${UNIT_PYTHON:?Set UNIT_PYTHON to the frozen unit environment Python.}"
eval_python="${EVAL_PYTHON:?Set EVAL_PYTHON to the frozen PI0 evaluator overlay Python.}"
artifact_root="${repo_root}/.local/artifacts/simulation/s4_3_pi1"
log_root="${repo_root}/.local/logs/simulation/s4_3_pi1/pi1d"
eval_root="${repo_root}/.local/experiments/simulation/s4_3_pi1/evaluation/seed1"
runtime_root="${repo_root}/.local/tmp/simulation/s4_3_pi1/pi1d"
contact_socket="${runtime_root}/contact_state.sock"
port=8131

if [[ ! "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "Invalid evaluation GPU: ${gpu_index}" >&2
  exit 2
fi
if [[ "$("${unit_python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${artifact_root}/pre_pi1d_freeze.json")" != PASS ]]; then
  echo "PI1D pre-evaluation freeze is not PASS." >&2
  exit 3
fi
if [[ -e "${eval_root}" || -e "${log_root}" || -e "${contact_socket}" ]]; then
  echo "Refusing to overwrite PI1D outputs, logs, or socket." >&2
  exit 4
fi

exec 9>"${common_git_dir}/tactile3d_unit_gpu${gpu_index}.lock"
if ! flock -n 9; then
  echo "Evaluation GPU ${gpu_index} lock is held." >&2
  exit 5
fi
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F, -v wanted="${gpu_index}" '{gsub(/ /, "", $1); gsub(/^ +| +$/, "", $2); if ($1 == wanted) print $2}')"
if [[ -z "${gpu_uuid}" ]] || nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fxq "${gpu_uuid}"; then
  echo "Evaluation GPU ${gpu_index} is busy after lock acquisition." >&2
  exit 6
fi
memory_used="$(nvidia-smi --id="${gpu_index}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( memory_used > 64 )); then
  echo "Evaluation GPU ${gpu_index} has ${memory_used} MiB allocated." >&2
  exit 6
fi

mkdir -p "${log_root}" "${eval_root}" "${runtime_root}"
contact_pid=""
policy_pid=""
stop_process() {
  local pid="$1"
  local label="$2"
  [[ -n "${pid}" ]] || return 0
  kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
  kill -INT "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 0.25
  done
  echo "${label} ignored SIGINT; sending SIGTERM." >&2
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || { wait "${pid}" 2>/dev/null || true; return 0; }
    sleep 0.25
  done
  echo "${label} did not stop after SIGTERM." >&2
  return 1
}
cleanup() {
  if [[ -n "${policy_pid}" ]] && kill -0 "${policy_pid}" 2>/dev/null; then
    stop_process "${policy_pid}" "policy server" || true
  fi
  if [[ -n "${contact_pid}" ]] && kill -0 "${contact_pid}" 2>/dev/null; then
    stop_process "${contact_pid}" "Contact-State service" || true
  fi
}
trap cleanup EXIT INT TERM

env PYTHONPATH="${repo_root}" PYTHONUNBUFFERED=1 \
  "${unit_python}" "${repo_root}/scripts/simulation/serve_s4_3_pi1_contact_state.py" \
  --socket "${contact_socket}" >"${log_root}/contact_state_service.log" 2>&1 &
contact_pid=$!
for _ in $(seq 1 120); do
  [[ -S "${contact_socket}" ]] && break
  kill -0 "${contact_pid}" 2>/dev/null || { tail -100 "${log_root}/contact_state_service.log" >&2; exit 7; }
  sleep 1
done
[[ -S "${contact_socket}" ]] || { echo "Contact-State service did not become ready." >&2; exit 7; }

declare -A modes=(
  [R0]=NONE
  [B0]=NONE
  [B1]=CONTACT_STATE_TOKENS
  [B2]=CONTACT_STATE_TOKENS_PHYSICAL_AUX
)
for model in R0 B0 B1 B2; do
  lower="${model,,}"
  echo "PI1D_MODEL_START model=${model} order=${lower}"
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_index}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    PYTHONUNBUFFERED=1 \
    "${openpi_python}" "${repo_root}/scripts/simulation/serve_s4_3_pi1_policy.py" \
    --model "${model}" --port "${port}" >"${log_root}/${lower}_server.log" 2>&1 &
  policy_pid=$!
  for _ in $(seq 1 180); do
    grep -q "PI1D_POLICY_SERVER_READY model=${model}" "${log_root}/${lower}_server.log" 2>/dev/null && break
    kill -0 "${policy_pid}" 2>/dev/null || { tail -120 "${log_root}/${lower}_server.log" >&2; exit 8; }
    sleep 1
  done
  grep -q "PI1D_POLICY_SERVER_READY model=${model}" "${log_root}/${lower}_server.log" || {
    echo "Policy server ${model} did not become ready." >&2
    exit 8
  }
  sleep 2
  env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${gpu_index}" MUJOCO_GL=egl \
    PYTHONUNBUFFERED=1 PYTHONPATH="${repo_root}:${repo_root}/third_party/dexjoco/dexjoco" \
    "${eval_python}" "${repo_root}/scripts/simulation/evaluate_s4_3_pi1d_augmented.py" \
    --model "${model}" --mode "${modes[$model]}" \
    --contact-socket "${contact_socket}" \
    --output "${eval_root}/${lower}" \
    --diagnostics-jsonl "${runtime_root}/${lower}_inference.jsonl" \
    --port "${port}" \
    --config "${repo_root}/third_party/dexjoco/configs/rand_obj/pinch_tongs.yaml" \
    --seed 1 --episodes 50 >"${log_root}/${lower}_client.log" 2>&1
  if [[ "$("${unit_python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${artifact_root}/pi1d_${lower}_eval.json")" != PASS ]]; then
    tail -150 "${log_root}/${lower}_client.log" >&2
    exit 9
  fi
  stop_process "${policy_pid}" "policy server ${model}"
  policy_pid=""
  echo "PI1D_MODEL_COMPLETE model=${model} result=$(grep -a 'Success rate:' "${log_root}/${lower}_client.log" | tail -1)"
done

stop_process "${contact_pid}" "Contact-State service"
contact_pid=""
if [[ "$("${unit_python}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${artifact_root}/contact_state_service.json")" != PASS ]]; then
  tail -150 "${log_root}/contact_state_service.log" >&2
  exit 10
fi
"${unit_python}" "${repo_root}/scripts/simulation/summarize_s4_3_pi1d.py"
echo "PI1D_EVALUATION_COMPLETE"
