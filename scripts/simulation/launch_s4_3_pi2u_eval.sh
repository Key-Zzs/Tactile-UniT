#!/usr/bin/env bash
# Launch fresh PI2U seed-6 evaluation on up to three genuinely idle GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${UNIT_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
PYTHON="${PYTHON:-python}"
ARTIFACT="$ROOT/.local/artifacts/simulation/s4_3_pi2u/pre_eval_retry_seed6.json"
LOG="$ROOT/.local/logs/simulation/s4_3_pi2u/evaluation_seed6_launcher.log"

test -f "$ARTIFACT"
test "$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$ARTIFACT")" = PASS

mapfile -t idle < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F, '
  NR == FNR {busy[$1] = 1; next}
  {gsub(/ /, "", $1); gsub(/ /, "", $2); if (($3 + 0) <= 64 && !busy[$2]) print $1}
' <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null) -)
chosen=()
for gpu in "${idle[@]}"; do
  chosen+=("$gpu")
  test "${#chosen[@]}" -eq 3 && break
done
test "${#chosen[@]}" -ge 1
GPUS="$(IFS=,; echo "${chosen[*]}")"

mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s s43_pi2u_eval_seed6 "cd '$ROOT' && '$PYTHON' scripts/simulation/run_s4_3_pi2u_eval.py launch --gpus '$GPUS' >> '$LOG' 2>&1"
echo "tmux=s43_pi2u_eval_seed6 gpus=$GPUS log=$LOG"
