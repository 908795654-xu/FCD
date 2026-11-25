#!/usr/bin/env bash
# Cross-dataset eval: load checkpoints from ETTh1/ETTh2 ablation, test on the other set.
# Usage: PRED_LEN=96 bash scripts/ablation/cross_dataset_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-/home/bkai/miniconda3/envs/forecast/bin/python}"
PRED_LEN="${PRED_LEN:-96}"
PL="pl${PRED_LEN}"
OUT="cross_dataset_results.txt"
if [ "$PRED_LEN" = "96" ]; then
  : >"$OUT"
else
  echo "" >>"$OUT"
  echo "===== pred_len=${PRED_LEN} =====" >>"$OUT"
fi

SEQ_LEN=512
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
)

ckpt_setting() {
  local train_data="$1" model_id="$2" des="$3"
  echo "${model_id}_PatchTST_${train_data}_ftM_sl${SEQ_LEN}_ll48_${PL}_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtFalse_${des}_0"
}

# variant: name dsacq_enable mem drift
VARIANTS=(
  "sacq|0|1|1"
  "drift|1|0|1"
  "mem|1|1|0"
  "full|1|1|1"
)

run_eval() {
  local gpu="$1" train_data="$2" test_data="$3" variant="$4"
  IFS='|' read -r vname dsacq mem drift <<<"$variant"
  local des model_id
  case "$vname" in
    sacq) des=SACQ_baseline; model_id="${train_data}_sacq_${PL}" ;;
    drift) des=DSACQ_drift_only; model_id="${train_data}_dsacq_drift_${PL}" ;;
    mem) des=DSACQ_mem_only; model_id="${train_data}_dsacq_mem_${PL}" ;;
    full) des=DSACQ_full; model_id="${train_data}_dsacq_full_${PL}" ;;
  esac
  local ckpt
  ckpt="$(ckpt_setting "$train_data" "$model_id" "$des")"
  if [ ! -f "./checkpoints/${ckpt}/checkpoint.pth" ]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 1
  fi
  local tag="${train_data}->${test_data}|${vname}|${PL}"
  local log="logs/LongForecasting/ablation/cross_${train_data}_to_${test_data}_${vname}_${PL}.log"
  echo "[GPU${gpu}] ${tag} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u run_longExp.py \
    "${COMMON[@]}" \
    --data "$test_data" \
    --data_path "${test_data}.csv" \
    --model_id "$model_id" \
    --des "$des" \
    --checkpoint_setting "$ckpt" \
    --eval_tag "$tag" \
    --dsacq_enable "$dsacq" \
    --dsacq_mem_gate_enable "$mem" \
    --dsacq_drift_gate_enable "$drift" \
    >"$log" 2>&1 &
}

echo "======== Cross-dataset eval pred_len=${PRED_LEN} ========"
idx=0
for train_data in ETTh1 ETTh2; do
  for test_data in ETTh1 ETTh2; do
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
cat "$OUT"
echo ""
echo "--- In-domain reference (from prior ablation logs) ---"
for d in ETTh1 ETTh2; do
  for v in sacq drift mem full; do
    f="logs/LongForecasting/ablation/${d}_${v}_${PL}.log"
    case "$v" in
      sacq) f="logs/LongForecasting/ablation/${d}_sacq_${PL}.log" ;;
      drift) f="logs/LongForecasting/ablation/${d}_dsacq_drift_${PL}.log" ;;
      mem) f="logs/LongForecasting/ablation/${d}_dsacq_mem_${PL}.log" ;;
      full) f="logs/LongForecasting/ablation/${d}_dsacq_full_${PL}.log" ;;
    esac
    if [ -f "$f" ]; then
      mse=$(grep -m1 '^mse:' "$f" || true)
      echo "${d}->${d} ${v}: ${mse}"
    fi
  done
done
