#!/usr/bin/env bash
# DSACQ ablation on 4 GPUs.
# Env: DATA (ETTh1|ETTh2), PRED_LEN (96|720|...), TRAIN_EPOCHS, PATIENCE
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
mkdir -p logs/LongForecasting/ablation

DATA="${DATA:-ETTh1}"
SEQ_LEN=512
PRED_LEN="${PRED_LEN:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
SEED=2021
PL_TAG="pl${PRED_LEN}"

COMMON=(
  --random_seed "$SEED"
  --is_training 1
  --root_path ./dataset/
  --data_path "${DATA}.csv"
  --model PatchTST
  --data "$DATA"
  --features M
  --seq_len "$SEQ_LEN"
  --pred_len "$PRED_LEN"
  --enc_in 7
  --e_layers 3
  --n_heads 4
  --d_model 16
  --d_ff 128
  --dropout 0.3
  --fc_dropout 0.3
  --head_dropout 0
  --patch_len 16
  --stride 8
  --head_type sacq_cross_attention
  --cross_attn_layers 2
  --sacq_alpha_mode learnable
  --itr 1
  --batch_size 128
  --learning_rate 0.0001
  --train_epochs "$TRAIN_EPOCHS"
  --patience "$PATIENCE"
)

run_one() {
  local gpu="$1"
  local model_id="$2"
  local des="$3"
  shift 3
  local log="logs/LongForecasting/ablation/${model_id}.log"
  echo "[${DATA} GPU${gpu}] pred_len=${PRED_LEN} ${des} -> ${log}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" \
    --model_id "$model_id" \
    --des "$des" \
    --gpu 0 \
    "$@" >"$log" 2>&1 &
}

run_one 0 "${DATA}_sacq_${PL_TAG}" SACQ_baseline --dsacq_enable 0
run_one 1 "${DATA}_dsacq_mem_${PL_TAG}" DSACQ_mem_only \
  --dsacq_enable 1 --dsacq_mem_gate_enable 1 --dsacq_drift_gate_enable 0
run_one 2 "${DATA}_dsacq_drift_${PL_TAG}" DSACQ_drift_only \
  --dsacq_enable 1 --dsacq_mem_gate_enable 0 --dsacq_drift_gate_enable 1
run_one 3 "${DATA}_dsacq_full_${PL_TAG}" DSACQ_full \
  --dsacq_enable 1 --dsacq_mem_gate_enable 1 --dsacq_drift_gate_enable 1

echo "Launched 4 jobs (${DATA}, pred_len=${PRED_LEN}). Waiting..."
wait
echo "--- ${DATA} pred_len=${PRED_LEN} test metrics ---"
grep -h "mse:" logs/LongForecasting/ablation/${DATA}_*_${PL_TAG}.log 2>/dev/null || true
