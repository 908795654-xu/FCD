#!/usr/bin/env bash
# ETTh2: DSACQ ablation at pred_len 96 then 720 (4 GPUs each round).
set -euo pipefail
export DATA=ETTh2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======== ETTh2 pred_len=96 ========"
PRED_LEN=96 bash "${SCRIPT_DIR}/dsacq_ablation_4gpu.sh"

echo "======== ETTh2 pred_len=720 ========"
PRED_LEN=720 bash "${SCRIPT_DIR}/dsacq_ablation_4gpu.sh"

echo "======== ETTh2 summary ========"
for pl in 96 720; do
  echo "--- pl${pl} ---"
  grep -h "mse:" "${SCRIPT_DIR}/../../logs/LongForecasting/ablation/ETTh2_"*"_pl${pl}.log" 2>/dev/null | sort
done
tail -20 "${SCRIPT_DIR}/../../result.txt" 2>/dev/null || true
