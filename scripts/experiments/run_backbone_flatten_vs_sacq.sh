#!/usr/bin/env bash
# 3 backbones × 3 datasets × 4 pred_lens × 2 heads — 72 tasks on 4 GPUs.
# Flatten+MSE vs SACQ (SAR+logcosh_scaled); epochs=30, patience=10.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/backbone_flatten_vs_sacq
mkdir -p "$OUT"

nohup "$PY" -u scripts/experiments/run_backbone_flatten_vs_sacq.py \
  --gpus 0,1,2,3 \
  --epochs 30 \
  --patience 10 \
  --out_dir "$OUT" \
  > "$OUT/master.log" 2>&1 &

echo "PID=$!"
echo "master log: $OUT/master.log"
echo "summary:    $OUT/summary.csv"
echo "tasks: 72 (3 models × 3 datasets × 4 pl × 2 variants)"
