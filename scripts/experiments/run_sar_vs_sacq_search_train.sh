#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs/etth1_sar_vs_sacq_search
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"

run_one() {
  local gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u run_longExp.py \
    --is_training 1 --random_seed 2021 \
    --model PatchTST \
    --root_path ./dataset/ --data ETTh1 --features M --freq h \
    --seq_len 512 --pred_len 720 --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
    --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
    --patch_len 16 --stride 8 \
    --train_epochs 50 --patience 20 --batch_size 128 --learning_rate 1e-4 \
    --lradj TST --pct_start 0.2 --itr 1 --num_workers 4 \
    --head_type sacq_cross_attention --sacq_scheme default \
    --sacq_alpha_mode learnable --short_setting 1 --skip_visual 1 \
    --dsacq_enable 0 --dsacq_mem_gate_enable 0 --dsacq_drift_gate_enable 0 \
    --loss logcosh_scaled --logcosh_beta 1.0 --logcosh_beta_mode auto_mad \
    "$@"
}

run_one 0 \
  --cross_attn_layers 2 \
  --data_path ETTh1_spike_p5.csv --scaler_data_path ETTh1.csv \
  --model_id H1s512p720SACQslc_spp5 --des hSACQslc_spp5 \
  --sacq_autoregressive 0 --sacq_pe_mode default \
  > logs/etth1_sar_vs_sacq_search/train_sacq_spp5.log 2>&1 &

run_one 1 \
  --cross_attn_layers 2 \
  --data_path ETTh1.csv \
  --model_id H1s512p720SACQ2lc --des hSACQ2lc \
  --sacq_autoregressive 0 --sacq_pe_mode relative_bias \
  > logs/etth1_sar_vs_sacq_search/train_sacq2lc.log 2>&1 &

run_one 2 \
  --cross_attn_layers 4 \
  --data_path ETTh1.csv \
  --model_id H1s512p720SARslc_L4 --des hSARslc_L4 \
  --sacq_autoregressive 1 --sacq_pe_mode default \
  > logs/etth1_sar_vs_sacq_search/train_sar_L4.log 2>&1 &

run_one 3 \
  --cross_attn_layers 4 \
  --data_path ETTh1.csv \
  --model_id H1s512p720SACQslc_L4 --des hSACQslc_L4 \
  --sacq_autoregressive 0 --sacq_pe_mode default \
  > logs/etth1_sar_vs_sacq_search/train_sacq_L4.log 2>&1 &

wait
echo "ALL_TRAIN_DONE"
