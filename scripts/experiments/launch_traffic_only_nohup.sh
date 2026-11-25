#!/usr/bin/env bash
# Traffic only: SACQ+SAR+logcosh, 4 GPUs parallel, nohup.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/batch_traffic_lga_queue
mkdir -p "$OUT"

nohup "$PY" -u scripts/experiments/gpu_queue_traffic_lga.py \
  --gpus 0,1,2,3 \
  --no-lga \
  > "$OUT/master_traffic.log" 2>&1 &

echo "PID=$!"
echo "master: $OUT/master_traffic.log"
echo "logs:   $OUT/traffic/"
