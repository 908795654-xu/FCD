#!/usr/bin/env bash
# LGA official defaults: ETTm1/ETTm2/Weather/Electricity × pl∈{96,192,336,720}, 4-GPU queue.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/batch_traffic_lga_queue
mkdir -p "$OUT/lga"

nohup "$PY" -u scripts/experiments/gpu_queue_traffic_lga.py \
  --gpus 0,1,2,3 \
  --no-traffic \
  > "$OUT/master_lga.log" 2>&1 &

echo "PID=$!"
echo "master: $OUT/master_lga.log"
echo "logs:   $OUT/lga/"
echo "tasks: 16 (4 datasets × 4 pred_len)"
