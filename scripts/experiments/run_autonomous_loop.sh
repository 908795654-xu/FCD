#!/usr/bin/env bash
# 持续运行 SACQ 偏移实验 sweep（用户离线时后台执行）
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT_DIR="${OUT_DIR:-logs/autonomous_sacq}"
GPUS="${GPUS:-0,1,2,3}"
PYTHON="/home/bkai/miniconda3/bin/conda run -n forecast python"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/artifacts"

echo "[autonomous] started at $(date -Iseconds)" | tee -a "$OUT_DIR/daemon.log"
while true; do
  echo "[autonomous] batch $(date -Iseconds)" | tee -a "$OUT_DIR/daemon.log"
  $PYTHON scripts/experiments/run_autonomous_sacq_shift.py \
    --out_dir "$OUT_DIR" \
    --gpus "$GPUS" \
    2>&1 | tee -a "$OUT_DIR/daemon.log" || true
  sleep 120
done
