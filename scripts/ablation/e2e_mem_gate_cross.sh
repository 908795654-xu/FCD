#!/usr/bin/env bash
# Cross-dataset eval for E2E mem gate models (ETTh1 <-> ETTh2, pl96/pl720).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
PRED_LEN="${PRED_LEN:-96}"
PL="pl${PRED_LEN}"
OUT="e2e_cross_dataset_results.txt"
SEQ_LEN=512

if [ "$PRED_LEN" = "96" ]; then
  : >"$OUT"
else
  echo "" >>"$OUT"
  echo "===== pred_len=${PRED_LEN} =====" >>"$OUT"
fi

COMMON=(
  --random_seed 2021
  --is_training 0
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
  --skip_visual 1
  --gpu 0
  --dsacq_enable 1
  --dsacq_mem_gate_enable 1
  --dsacq_mem_gate_mode e2e
)

ckpt_setting() {
  local train_data="$1" model_id="$2" des="$3"
  echo "${model_id}_PatchTST_${train_data}_ftM_sl${SEQ_LEN}_ll48_${PL}_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_${des}_0"
}

VARIANTS=(
  "e2e_mem|0|DSACQ_e2e_mem"
  "e2e_full|1|DSACQ_e2e_full"
)

run_eval() {
  local gpu="$1" train_data="$2" test_data="$3" variant="$4"
  IFS='|' read -r vname drift des <<<"$variant"
  local model_id="${train_data}_dsacq_${vname}_${PL}"
  local ckpt
  ckpt="$(ckpt_setting "$train_data" "$model_id" "$des")"
  if [ ! -f "./checkpoints/${ckpt}/checkpoint.pth" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 1
  fi
  local tag="${train_data}->${test_data}|${vname}|${PL}"
  local log="logs/LongForecasting/ablation/cross_e2e_${train_data}_to_${test_data}_${vname}_${PL}.log"
  echo "[GPU${gpu}] ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" \
    --data "$test_data" \
    --data_path "${test_data}.csv" \
    --model_id "$model_id" \
    --des "$des" \
    --checkpoint_setting "$ckpt" \
    --eval_tag "$tag" \
    --dsacq_drift_gate_enable "$drift" \
    >"$log" 2>&1 &
}

echo "======== E2E cross-dataset eval pred_len=${PRED_LEN} ========"
idx=0
for train_data in ETTh1 ETTh2; do
  for test_data in ETTh1 ETTh2; do
    if [ "$train_data" = "$test_data" ]; then
      continue
    fi
    for variant in "${VARIANTS[@]}"; do
      gpu=$((idx % 4))
      run_eval "$gpu" "$train_data" "$test_data" "$variant"
      idx=$((idx + 1))
      if (( idx % 4 == 0 )); then
        wait
      fi
    done
  done
done
wait

echo "======== Summary (${OUT}) ========"
grep -h "mse:" logs/LongForecasting/ablation/cross_e2e_*_${PL}.log 2>/dev/null | sort || true
for f in logs/LongForecasting/ablation/cross_e2e_*_${PL}.log; do
  [ -f "$f" ] || continue
  tag=$(grep -m1 'eval_tag=' "$f" | sed "s/.*eval_tag='\\([^']*\\)'.*/\\1/" || basename "$f")
  mse=$(grep -m1 '^mse:' "$f" || echo "missing")
  echo "${tag}: ${mse}" >>"$OUT"
done
cat "$OUT"
