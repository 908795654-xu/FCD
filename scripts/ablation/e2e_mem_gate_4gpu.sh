#!/usr/bin/env bash
# E2E memory gate vs full DSACQ (e2e mem + drift). 4 GPUs per round.
# Env: DATA=ETTh1|ETTh2, PRED_LEN=96|720
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
mkdir -p logs/LongForecasting/ablation

DATA="${DATA:-ETTh1}"
PRED_LEN="${PRED_LEN:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
GPU_OFFSET="${GPU_OFFSET:-0}"
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
  --seq_len 512
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
  --dsacq_enable 1
  --dsacq_mem_gate_enable 1
  --dsacq_mem_gate_mode e2e
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

# Only 2 variants per dataset round; caller sets GPU_OFFSET (0 or 2) for 4-GPU layout.
run_one "$((GPU_OFFSET + 0))" "${DATA}_dsacq_e2e_mem_${PL_TAG}" DSACQ_e2e_mem \
  --dsacq_drift_gate_enable 0
run_one "$((GPU_OFFSET + 1))" "${DATA}_dsacq_e2e_full_${PL_TAG}" DSACQ_e2e_full \
  --dsacq_drift_gate_enable 1

wait
echo "--- ${DATA} pred_len=${PRED_LEN} ---"
grep -h "mse:" logs/LongForecasting/ablation/${DATA}_dsacq_e2e_*_${PL_TAG}.log 2>/dev/null || true
