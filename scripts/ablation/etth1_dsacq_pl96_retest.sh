#!/usr/bin/env bash
# Re-run test only for completed ablation checkpoints.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"

COMMON=(
  --random_seed 2021
  --is_training 0
  --root_path ./dataset/
  --data_path ETTh1.csv
  --model PatchTST
  --data ETTh1
  --features M
  --seq_len 512
  --pred_len 96
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
)

test_one() {
  local gpu="$1"
  local model_id="$2"
  local des="$3"
  shift 3
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" --model_id "$model_id" --des "$des" --gpu 0 "$@" \
    >>"logs/LongForecasting/ablation/${model_id}_test.log" 2>&1 &
}

mkdir -p logs/LongForecasting/ablation
test_one 0 ETTh1_sacq_pl96 SACQ_baseline --dsacq_enable 0
test_one 1 ETTh1_dsacq_mem_pl96 DSACQ_mem_only --dsacq_enable 1 --dsacq_mem_gate_enable 1 --dsacq_drift_gate_enable 0
test_one 2 ETTh1_dsacq_drift_pl96 DSACQ_drift_only --dsacq_enable 1 --dsacq_mem_gate_enable 0 --dsacq_drift_gate_enable 1
test_one 3 ETTh1_dsacq_full_pl96 DSACQ_full --dsacq_enable 1 --dsacq_mem_gate_enable 1 --dsacq_drift_gate_enable 1
wait
grep -h "mse:" logs/LongForecasting/ablation/*_test.log
tail -12 result.txt 2>/dev/null || true
