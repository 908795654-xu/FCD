#!/usr/bin/env bash
# Traffic (SACQ+SAR+logcosh, traffic.sh) + LGA official 4 datasets; 4-GPU queue, nohup.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/batch_traffic_lga_queue
mkdir -p "$OUT"

nohup "$PY" -u scripts/experiments/gpu_queue_traffic_lga.py \
  --gpus 0,1,2,3 \
  > "$OUT/master.log" 2>&1 &

echo "PID=$!"
echo "master: $OUT/master.log"
echo "tasks: traffic×4 (SAR+logcosh_scaled) + LGA×16 (ETTm1/2/Weather/Electricity)"
