#!/usr/bin/env bash
# Traffic + Electricity: SAR + logcosh_scaled, 4 GPUs, pl=96/192/336/720
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/traffic_electricity_sar_lc_scaled
mkdir -p "$OUT"

nohup "$PY" -u scripts/experiments/run_traffic_electricity_sar_lc_scaled.py \
  --gpus 0,1,2,3 \
  --datasets traffic,electricity \
  --out_dir "$OUT" \
  > "$OUT/master.log" 2>&1 &

echo "PID=$!"
echo "master log: $OUT/master.log"
echo "tasks: 8 (traffic×4 + electricity×4 pred_len)"
