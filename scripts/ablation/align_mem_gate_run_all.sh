#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

run_pl() {
  local pl="$1"
  echo "======== pred_len=${pl}: ETTh1 (GPU0-1) + ETTh2 (GPU2-3) ========"
  DATA=ETTh1 PRED_LEN="$pl" GPU_OFFSET=0 bash "${SCRIPT_DIR}/align_mem_gate_4gpu.sh" &
  pid1=$!
  DATA=ETTh2 PRED_LEN="$pl" GPU_OFFSET=2 bash "${SCRIPT_DIR}/align_mem_gate_4gpu.sh" &
  pid2=$!
  wait "$pid1" "$pid2"
}

run_pl 96
run_pl 720

echo "======== In-domain summary (align) ========"
for d in ETTh1 ETTh2; do
  for pl in 96 720; do
    echo "--- ${d} pl${pl} ---"
    for v in align_mem align_full; do
      f="logs/LongForecasting/ablation/${d}_dsacq_${v}_pl${pl}.log"
      [ -f "$f" ] && grep -m1 '^mse:' "$f" || echo "missing $f"
    done
  done
done
