#!/usr/bin/env bash
# ETTh1 sl512 pl96: train missing checkpoints + test corruption sweeps (GPU parallel).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
export PYTHON="$PY"

OUT="logs/etth1_pl720_seqlen_sweep/train_corruption_sl512_pl96"
CKPT_FLATSLC="checkpoints/ETTh1_512_96_PatchTST_ETTh1_sl512_pl96_hFlatslc_0/checkpoint.pth"

mkdir -p "$OUT"

if [[ ! -f "$CKPT_FLATSLC" ]]; then
  echo "=== train Flatten+logcosh (pl96, sl512) on GPU 0 ==="
  "$PY" scripts/experiments/run_etth1_pl720_seqlen_4way.py \
    --seq_lens 512 --pred_len 96 --gpus 0 \
    --variants flatten_logcosh_scaled \
    --out_dir logs/etth1_pl720_seqlen_sweep/train_pl96_flat_slc
else
  echo "=== skip train: $CKPT_FLATSLC exists ==="
fi

echo "=== parallel: mask_sweep (GPU0) + spike_sweep (GPU1) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/experiments/eval_etth1_train_corruption_sl512.py \
  --pred_len 96 --split test --mode mask_sweep --gpu 0 &
PID_MASK=$!
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/experiments/eval_etth1_train_corruption_sl512.py \
  --pred_len 96 --split test --mode spike_sweep --gpu 0 &
PID_SPIKE=$!
wait "$PID_MASK" "$PID_SPIKE"

echo "=== done -> $OUT ==="
ls -la "$OUT"/test_corruption_mask_mse.png "$OUT"/test_corruption_spike_mse.png
