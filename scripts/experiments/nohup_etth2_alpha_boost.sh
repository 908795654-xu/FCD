#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="${ROOT}/logs/etth2_alpha_lr_boost"
mkdir -p "$OUT"
PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
export PYTHON
LOG="${OUT}/launcher.log"
echo "start $(date -Iseconds)" | tee "$LOG"
nohup "$PYTHON" -u scripts/experiments/run_etth2_alpha_lr_boost.py \
  --gpus 0,1,2,3 \
  --out_dir logs/etth2_alpha_lr_boost \
  --alpha_ratios 2,5,10 \
  >> "$LOG" 2>&1 &
echo "pid=$! log=$LOG" | tee -a "$LOG"
