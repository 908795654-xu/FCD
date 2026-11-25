#!/usr/bin/env bash
# ETT×4: Linear / DLinear / PatchTSTMamba × Flatten+MSE vs SAR+logcosh_scaled
# Phase 1: pred_len 96,720 — Phase 2: pred_len 192,336 (after phase 1)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/ett4_backbone_flatten_vs_sar
mkdir -p "$OUT"

run_phase() {
  local phase="$1"
  local pls="$2"
  echo "[$(date '+%F %T')] === phase ${phase}: pred_lens=${pls} ===" | tee -a "$OUT/master.log"
  "$PY" -u scripts/experiments/run_backbone_flatten_vs_sacq.py \
    --gpus 0,1,2,3 \
    --datasets ETTh1,ETTh2,ETTm1,ETTm2 \
    --models PatchTSTMamba,Linear,DLinear \
    --pred_lens "$pls" \
    --epochs 30 \
    --patience 10 \
    --out_dir "$OUT" \
    2>&1 | tee -a "$OUT/master.log"
}

case "${1:-all}" in
  phase1)
    run_phase 1 "96,720"
    ;;
  phase2)
    run_phase 2 "192,336"
    ;;
  all)
    run_phase 1 "96,720"
    run_phase 2 "192,336"
    ;;
  *)
    echo "Usage: $0 [phase1|phase2|all]"
    exit 1
    ;;
esac

echo "[$(date '+%F %T')] ALL_DONE" | tee -a "$OUT/master.log"
