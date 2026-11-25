#!/usr/bin/env bash
# ETTh1 pl=720: sl x L grid, SAR+logcosh_scaled (ref H1s512p720SARslc protocol)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/etth1_sl_L_grid_pl720
mkdir -p "$OUT/logs"

nohup "$PY" -u scripts/experiments/run_etth1_sl_L_grid.py \
  --gpus 0,1,2,3 \
  --out_dir "$OUT" \
  > "$OUT/master.log" 2>&1 &

echo "PID=$!"
echo "master log: $OUT/master.log"
echo "tasks: 15 (5 seq_len x 3 L), pl=720, logcosh_scaled k=1 auto_mad"
