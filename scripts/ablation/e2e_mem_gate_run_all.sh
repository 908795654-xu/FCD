#!/usr/bin/env bash
# Train E2E mem gate experiments: ETTh1/ETTh2 × pl96/pl720 (4-GPU parallel per pred_len).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_pl() {
  local pl="$1"
  echo "======== pred_len=${pl}: ETTh1 (GPU0-1) + ETTh2 (GPU2-3) ========"
  DATA=ETTh1 PRED_LEN="$pl" GPU_OFFSET=0 bash "${SCRIPT_DIR}/e2e_mem_gate_4gpu.sh" &
  pid1=$!
  DATA=ETTh2 PRED_LEN="$pl" GPU_OFFSET=2 bash "${SCRIPT_DIR}/e2e_mem_gate_4gpu.sh" &
  pid2=$!
  wait "$pid1" "$pid2"
}

run_pl 96
run_pl 720

echo "======== In-domain summary ========"
for d in ETTh1 ETTh2; do
  for pl in 96 720; do
    echo "--- ${d} pl${pl} ---"
    for v in e2e_mem e2e_full; do
      f="logs/LongForecasting/ablation/${d}_dsacq_${v}_pl${pl}.log"
      if [ -f "$f" ]; then
        grep -m1 '^mse:' "$f" || true
      fi
    done
  done
done
