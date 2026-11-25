#!/usr/bin/env bash
# ETTh1 + Weather: L∈{1,2,3} × k∈{0.5,1,2,4}, pl=720, SAR+logcosh_scaled (auto_mad)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/sacq_Lk_sensitivity
mkdir -p "$OUT"

nohup "$PY" -u scripts/experiments/run_sacq_Lk_sensitivity.py \
  --gpus 0,1,2,3 \
  --datasets ETTh1,weather \
  --out_dir "$OUT" \
  > "$OUT/master.log" 2>&1 &

echo "PID=$!"
echo "master log: $OUT/master.log"
echo "tasks: 24 (2 datasets × 3 L × 4 k), pl=720"
