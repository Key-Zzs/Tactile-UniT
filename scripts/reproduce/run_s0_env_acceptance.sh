#!/usr/bin/env bash
# Run the S0 environment checks incrementally and retain a log for every gate.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/.local/logs/setup/s0_env"
mkdir -p "$LOG_DIR"

CONDA_SH="${CONDA_SH:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unit}"
if [[ -z "$CONDA_SH" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "FAIL: conda is not on PATH; set CONDA_SH explicitly" >&2
    exit 2
  fi
  CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
if [[ ! -r "$CONDA_SH" ]]; then
  echo "FAIL: conda initialization script not found: $CONDA_SH" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

declare -A RESULT
run_gate() {
  local key="$1" logfile="$2"; shift 2
  echo "[$key] $*" | tee "$LOG_DIR/$logfile"
  if "$@" >>"$LOG_DIR/$logfile" 2>&1; then
    RESULT["$key"]="PASS"
    echo "[$key] PASS" | tee -a "$LOG_DIR/$logfile"
  else
    RESULT["$key"]="FAIL"
    echo "[$key] FAIL (traceback retained in $LOG_DIR/$logfile)" | tee -a "$LOG_DIR/$logfile"
  fi
}

run_audit() {
  {
    pwd; git status --short; git branch --show-current; git rev-parse HEAD; git remote -v
    hostname; uname -a; cat /etc/os-release
    nvidia-smi; nvidia-smi topo -m
    command -v nvcc || true; nvcc --version || true
    command -v conda || true; conda env list
    printf 'PATH=%s\nLD_LIBRARY_PATH=%s\nDISPLAY=%s\nMUJOCO_GL=%s\nPYOPENGL_PLATFORM=%s\n' \
      "$PATH" "${LD_LIBRARY_PATH:-}" "${DISPLAY:-<unset>}" "${MUJOCO_GL:-<unset>}" "${PYOPENGL_PLATFORM:-<unset>}"
    python --version; command -v python
    ldconfig -p | grep -E 'libEGL|libGLX|libOpenGL|libOSMesa' || true
    ls -la /usr/share/glvnd/egl_vendor.d 2>/dev/null || true
  } >"$LOG_DIR/hardware.log" 2>&1
}
if run_audit; then RESULT[A0]="PASS"; else RESULT[A0]="FAIL"; fi

run_gate A1 training_env.log python scripts/reproduce/check_training_env.py --stage imports
run_gate A2 cuda.log python scripts/reproduce/check_training_env.py --stage cuda
run_gate A2_FLASH flash_attention.log python scripts/reproduce/check_training_env.py --stage flash-attn

# robosuite 1.5.1 checks that EGL ID is one of CUDA_VISIBLE_DEVICES. The default
# follows the single-GPU-0 smoke-test policy; choose another *matching physical*
# ID only when GPU 0 is unavailable, e.g. S0_SIM_CUDA_VISIBLE_DEVICES=3
# S0_MUJOCO_EGL_DEVICE_ID=3.
unset DISPLAY
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${S0_SIM_CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${S0_MUJOCO_EGL_DEVICE_ID:-${S0_SIM_CUDA_VISIBLE_DEVICES:-0}}"
run_gate A3 mujoco_physics.log python scripts/reproduce/check_headless_sim.py --stage mujoco-physics
run_gate A4 mujoco_egl.log python scripts/reproduce/check_headless_sim.py --stage mujoco-egl
run_gate A5 robosuite.log python scripts/reproduce/check_headless_sim.py --stage robosuite
run_gate A6 robocasa.log python scripts/reproduce/check_headless_sim.py --stage robocasa

CHECKPOINT="${UNIT_CHECKPOINT_PATH:-}"
if [[ -n "$CHECKPOINT" && -d "$CHECKPOINT" && -f "$CHECKPOINT/config.json" ]]; then
  run_gate A7 unit_eval_smoke.log env N_ENVS=1 N_EPISODES=2 bash examples/run_eval.sh "$CHECKPOINT" id
else
  RESULT[A7]="SKIP"
  {
    echo "SKIP — UNIT_CHECKPOINT_PATH was not supplied; no evaluation was run"
    echo "Exact command when ready: unset DISPLAY; MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID} N_ENVS=1 N_EPISODES=2 bash examples/run_eval.sh <checkpoint_dir> id"
  } >"$LOG_DIR/unit_eval_smoke.log"
fi

echo "================================"
echo "S0 Environment Acceptance"
echo "================================"
for key in A0 A1 A2 A2_FLASH A3 A4 A5 A6 A7; do printf '%-20s %s\n' "$key" "${RESULT[$key]}"; done
mandatory=(A0 A1 A2 A2_FLASH A3 A4 A5 A6)
passed=0
for key in "${mandatory[@]}"; do [[ "${RESULT[$key]}" == PASS ]] && ((passed+=1)); done
echo "--------------------------------"
echo "MANDATORY: $passed/${#mandatory[@]} PASS"
echo "OPTIONAL : ${RESULT[A7]}"
echo "================================"
[[ "$passed" == "${#mandatory[@]}" ]]
