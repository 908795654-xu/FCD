#!/usr/bin/env bash
# Wait for current electricity batch to finish, then auto-restart traffic only.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/traffic_electricity_sar_lc_scaled
MASTER_PID="${1:-1227751}"
LOG="$OUT/traffic_after_ele.log"

echo "[$(date '+%F %T')] waiting for master PID=$MASTER_PID to finish (electricity batch)..." | tee -a "$LOG"

while kill -0 "$MASTER_PID" 2>/dev/null; do
  sleep 120
done

echo "[$(date '+%F %T')] master done. sleeping 30s for GPU cleanup..." | tee -a "$LOG"
sleep 30

echo "[$(date '+%F %T')] starting traffic×4 (SAR + logcosh_scaled) on GPUs 0,1,2,3..." | tee -a "$LOG"
"$PY" -u scripts/experiments/run_traffic_electricity_sar_lc_scaled.py \
  --gpus 0,1,2,3 \
  --datasets traffic \
  --out_dir "$OUT" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] traffic batch ALL_DONE" | tee -a "$LOG"
