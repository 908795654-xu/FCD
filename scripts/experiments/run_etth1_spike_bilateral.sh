#!/usr/bin/env bash
# ETTh1 pl=720: bilateral (train+test CSV) spike experiment.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"

echo "=== 1) Generate bilateral CSVs ==="
"$PY" scripts/experiments/run_etth1_spike_bilateral_pl720.py --prepare_only --force_data

echo "=== 2) Train SAR / SACQ / Flatten (0% and 5% bilateral) ==="
"$PY" scripts/experiments/run_etth1_spike_bilateral_pl720.py \
  --gpus "${GPUS:-0,1,2}" \
  --outlier_pcts "${PCTS:-0,5}" \
  --noise_mult "${NOISE_MULT:-3}" \
  "$@"

echo "=== 3) Plot ==="
"$PY" scripts/experiments/plot_etth1_spike_bilateral_pl720.py

echo "DONE -> logs/etth1_spike_bilateral_pl720/"
