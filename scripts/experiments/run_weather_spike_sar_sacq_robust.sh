#!/usr/bin/env bash
# Weather: spike-train SAR/SACQ + robustness figures (fast train).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
OUT=logs/weather_sar_vs_sacq_search
mkdir -p "$OUT"

echo "== prepare weather spike 5% ±3σ =="
$PY -c "
from scripts.experiments.outlier_utils import inject_spike_outliers
import os
src='./dataset/weather.csv'
dst='./dataset/weather_spike_3x_p5.csv'
if not os.path.isfile(dst):
    meta=inject_spike_outliers(src,dst,outlier_ratio=0.05,noise_mult=3.0,train_rows=36887,seed=2021)
    print('created', dst, meta['n_outliers'])
else:
    print('exists', dst)
"

train_one() {
  local gpu=$1 ar=$2 mid=$3 des=$4 loss=$5
  local ckpt="checkpoints/${mid}_PatchTST_weather_sl512_pl720_${des}_0/checkpoint.pth"
  if [[ -f "$ckpt" ]]; then
    echo "skip train (exists) $mid"
    return 0
  fi
  echo "train $mid on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u run_longExp.py \
    --is_training 1 --random_seed 2021 --model PatchTST \
    --root_path ./dataset/ --data_path weather_spike_3x_p5.csv --scaler_data_path weather.csv \
    --data weather --features M --freq h --seq_len 512 --pred_len 720 \
    --enc_in 21 --dec_in 21 --c_out 21 --e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 \
    --dropout 0.2 --fc_dropout 0.2 --head_dropout 0 --patch_len 16 --stride 8 \
    --cross_attn_layers 2 --train_epochs 20 --patience 8 --batch_size 128 --learning_rate 1e-4 \
    --lradj TST --pct_start 0.2 --itr 1 --num_workers 4 \
    --head_type sacq_cross_attention --sacq_scheme default --sacq_pe_mode default \
    --sacq_autoregressive $ar --sacq_alpha_mode learnable --short_setting 1 --skip_visual 1 \
    --dsacq_enable 0 --dsacq_mem_gate_enable 0 --dsacq_drift_gate_enable 0 \
    --loss $loss --logcosh_beta 1.0 --logcosh_beta_mode auto_mad \
    --model_id $mid --des $des \
    > "$OUT/train_${des}.log" 2>&1 &
}

train_one 0 1 Ws512p720SARlc_spp5 hSARlc_spp5 logcosh
train_one 1 0 Ws512p720SACQslc_spp5 hSACQslc_spp5 logcosh_scaled
wait
echo "TRAIN_DONE"

$PY scripts/experiments/eval_sar_vs_sacq_spike_train_shift.py \
  --dataset weather --out_dir "$OUT" --pair spike \
  --shift_max 10 --shift_step 2 --shift_min 0 --no_all \
  --fig_name fig_spike_train_tail_shift_noall.png

$PY scripts/experiments/plot_sar_vs_sacq_spike_horizon_mse.py \
  --dataset weather --out_dir "$OUT" \
  --shift_pcts 5,10 --bin_size 72 \
  --fig_name fig_spike_train_horizon_mse.png

echo "ALL_DONE -> $OUT"
