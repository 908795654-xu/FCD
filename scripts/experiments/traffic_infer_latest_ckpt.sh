#!/usr/bin/env bash
# Load latest checkpoint.pth and run test-set forward; append to traffic logs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
LOG_DIR="logs/batch_traffic_lga_queue/traffic"

run_infer() {
  local pl="$1"
  local gpu="$2"
  local log="${LOG_DIR}/traffic_512_${pl}_SARslc.log"
  local model_id="traffic_512_${pl}_SARslc"
  local setting="${model_id}_PatchTST_traffic_sl512_pl${pl}_Exp_0"
  local ckpt="checkpoints/${setting}/checkpoint.pth"

  {
    echo ""
    echo "========== INFERENCE (latest checkpoint) $(date '+%F %T') =========="
    echo "setting=${setting}"
    echo "checkpoint=${ckpt}"
    ls -la "$ckpt" 2>&1 || true
    echo "---"
  } >> "$log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u run_longExp.py \
    --random_seed 2021 \
    --is_training 0 \
    --root_path "$ROOT/dataset/" \
    --data_path traffic.csv \
    --model_id "$model_id" \
    --model PatchTST \
    --data traffic \
    --features M \
    --seq_len 512 \
    --pred_len "$pl" \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0.05 \
    --patch_len 16 --stride 8 --cross_attn_layers 1 \
    --sacq_coarse_lr_ratio 1.0 --sacq_decoder_lr_ratio 1.0 \
    --sacq_alpha_lr_ratio 1.0 --sacq_backbone_lr_ratio 0.5 \
    --head_type sacq_cross_attention \
    --sacq_autoregressive 1 --sacq_alpha_mode learnable \
    --short_setting 1 --des Exp --skip_visual 1 \
    --dsacq_enable 0 --dsacq_mem_gate_enable 0 --dsacq_drift_gate_enable 0 \
    --loss logcosh_scaled --logcosh_beta 1.0 --logcosh_beta_mode auto_mad \
    --batch_size 4 --num_workers 0 \
    --gpu 0 \
    >> "$log" 2>&1

  echo "INFERENCE_DONE pl=${pl} gpu=${gpu}" >> "$log"
}

# 顺序执行，避免并行占满 GPU / shm
for pl_gpu in "96:0" "192:1" "336:2" "720:3"; do
  pl="${pl_gpu%%:*}"
  gpu="${pl_gpu##*:}"
  run_infer "$pl" "$gpu"
done
echo "ALL_INFER_DONE"
