#!/usr/bin/env bash
# Align 5-component ablation on ETTh1 pl96, cross-eval ETTh1->ETTh2. 4 GPUs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
mkdir -p logs/LongForecasting/ablation
OUT="align_component_ablation_etth1_to_etth2_pl96.txt"
: >"$OUT"

PRED_LEN=96
PL="pl${PRED_LEN}"
SEQ_LEN=512
SEED=2021

COMMON=(
  --random_seed "$SEED"
  --root_path ./dataset/
  --model PatchTST
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
  --train_epochs 30
  --patience 10
  --dsacq_enable 1
  --dsacq_mem_gate_enable 1
  --dsacq_drift_gate_enable 1
  --dsacq_mem_gate_mode align
  --mem_shift_aug_std 0.15
  --mem_shift_aug_prob 0.5
  --mem_coral_loss_weight 0.05
)

train_one() {
  local gpu="$1" model_id="$2" des="$3"
  shift 3
  local log="logs/LongForecasting/ablation/${model_id}.log"
  echo "[TRAIN GPU${gpu}] ${des} -> ${log}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" \
    --is_training 1 \
    --data ETTh1 \
    --data_path ETTh1.csv \
    --model_id "$model_id" \
    --des "$des" \
    --gpu 0 \
    "$@" >"$log" 2>&1 &
}

cross_one() {
  local gpu="$1" model_id="$2" des="$3" tag="$4"
  shift 4
  local ckpt="${model_id}_PatchTST_ETTh1_ftM_sl${SEQ_LEN}_ll48_${PL}_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_${des}_0"
  local log="logs/LongForecasting/ablation/cross_${model_id}_to_ETTh2.log"
  echo "[CROSS GPU${gpu}] ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" \
    --is_training 0 \
    --data ETTh2 \
    --data_path ETTh2.csv \
    --model_id "$model_id" \
    --des "$des" \
    --checkpoint_setting "$ckpt" \
    --eval_tag "$tag" \
    --skip_visual 1 \
    --gpu 0 \
    "$@" >"$log" 2>&1 &
}

echo "======== Train 5 ablation variants (full already trained) ========"
train_one 0 ETTh1_align_ab_noshift_pl96 AlignAb_noShift --align_ablate_shift 1
train_one 1 ETTh1_align_ab_nodual_pl96 AlignAb_noDualBank --align_ablate_dual_bank 1
train_one 2 ETTh1_align_ab_notoken_pl96 AlignAb_noTokenGate --align_ablate_token_gate 1
train_one 3 ETTh1_align_ab_nodrift_pl96 AlignAb_noShiftDrift --align_ablate_shift_drift 1
wait
train_one 0 ETTh1_align_ab_notrain_pl96 AlignAb_noTrainReg \
  --mem_shift_aug_prob 0 --mem_coral_loss_weight 0
wait

echo "======== Cross-eval ETTh1 -> ETTh2 ========"
cross_one 0 ETTh1_dsacq_align_full_pl96 DSACQ_align_full "ETTh1->ETTh2|full|pl96"
cross_one 1 ETTh1_align_ab_noshift_pl96 AlignAb_noShift "ETTh1->ETTh2|ab_no_shift|pl96" --align_ablate_shift 1
cross_one 2 ETTh1_align_ab_nodual_pl96 AlignAb_noDualBank "ETTh1->ETTh2|ab_no_dual|pl96" --align_ablate_dual_bank 1
cross_one 3 ETTh1_align_ab_notoken_pl96 AlignAb_noTokenGate "ETTh1->ETTh2|ab_no_token|pl96" --align_ablate_token_gate 1
wait
cross_one 0 ETTh1_align_ab_nodrift_pl96 AlignAb_noShiftDrift "ETTh1->ETTh2|ab_no_drift|pl96" --align_ablate_shift_drift 1
cross_one 1 ETTh1_align_ab_notrain_pl96 AlignAb_noTrainReg "ETTh1->ETTh2|ab_no_train|pl96" \
  --mem_shift_aug_prob 0 --mem_coral_loss_weight 0
wait

{
  echo "===== Align component ablation: ETTh1 -> ETTh2 pl96 ====="
  declare -A MAP=(
    [full]="logs/LongForecasting/ablation/cross_ETTh1_dsacq_align_full_pl96_to_ETTh2.log|完整 align_full"
    [no_shift]="logs/LongForecasting/ablation/cross_ETTh1_align_ab_noshift_pl96_to_ETTh2.log|w/o ①漂移编码"
    [no_dual]="logs/LongForecasting/ablation/cross_ETTh1_align_ab_nodual_pl96_to_ETTh2.log|w/o ②双记忆库路由"
    [no_token]="logs/LongForecasting/ablation/cross_ETTh1_align_ab_notoken_pl96_to_ETTh2.log|w/o ③Query token门控"
    [no_drift]="logs/LongForecasting/ablation/cross_ETTh1_align_ab_nodrift_pl96_to_ETTh2.log|w/o ④统一漂移残差门控"
    [no_train]="logs/LongForecasting/ablation/cross_ETTh1_align_ab_notrain_pl96_to_ETTh2.log|w/o ⑤训练增强+CORAL"
  )
  for key in full no_shift no_dual no_token no_drift no_train; do
    IFS='|' read -r log label <<<"${MAP[$key]}"
    mse=$(grep -m1 '^mse:' "$log" 2>/dev/null || echo "missing")
    echo "${label}: ${mse}"
  done
} | tee "$OUT"
