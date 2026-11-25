"""Evaluate Weather SACQ checkpoints using only the coarse branch (sacq_flat_seq)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main
from utils.metrics import metric

PRED_LENS = [96, 192, 336, 720]

# Configs match pths&logs/logs/Weather/*.log
WEATHER_RUNS = {
    96: {
        "model_id": "Ws512p96SARslc",
        "setting": "Ws512p96SARslc_PatchTST_weather_sl512_pl96_wSARslc_0",
        "cross_attn_layers": 2,
        "head_dropout": 0.0,
        "des": "wSARslc",
        "logcosh_beta": 1.0,
        "full_mse": 0.14688237011432648,
        "full_mae": 0.18519560992717743,
    },
    192: {
        "model_id": "Ws512p192SARslc",
        "setting": "Ws512p192SARslc_PatchTST_weather_sl512_pl192_wSARslc_0",
        "cross_attn_layers": 2,
        "head_dropout": 0.0,
        "des": "wSARslc",
        "logcosh_beta": 1.0,
        "full_mse": 0.1929798126220703,
        "full_mae": 0.22968430817127228,
    },
    336: {
        "model_id": "Ws512p336SARslc",
        "setting": "Ws512p336SARslc_PatchTST_weather_sl512_pl336_wSARslc_0",
        "cross_attn_layers": 2,
        "head_dropout": 0.0,
        "des": "wSARslc",
        "logcosh_beta": 1.0,
        "full_mse": 0.24307453632354736,
        "full_mae": 0.2708144783973694,
    },
    720: {
        "model_id": "Ws512p720SAR_L1k4",
        "setting": "Ws512p720SAR_L1k4_PatchTST_weather_sl512_pl720_hSAR_L1k4_0",
        "cross_attn_layers": 1,
        "head_dropout": 0.0,
        "des": "hSAR_L1k4",
        "logcosh_beta": 4.0,
        "full_mse": 0.3103117346763611,
        "full_mae": 0.3259959816932678,
    },
}

DEFAULT_CKPT_DIR = os.path.join("pths&logs", "pths", "Weather")


def make_weather_args(pred_len: int, gpu: int = 0, use_gpu: bool = True) -> Namespace:
    run = WEATHER_RUNS[pred_len]
    return Namespace(
        random_seed=2021,
        is_training=0,
        model_id=run["model_id"],
        model="PatchTST",
        data="weather",
        root_path="./dataset/",
        data_path="weather.csv",
        scaler_data_path="",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        checkpoint_setting=run["setting"],
        init_checkpoint_setting="",
        skip_visual=1,
        eval_tag="coarse_only",
        seq_len=512,
        label_len=48,
        pred_len=pred_len,
        fc_dropout=0.2,
        head_dropout=run["head_dropout"],
        patch_len=16,
        stride=8,
        head_type="sacq_cross_attention",
        dsacq_coarse_split=1,
        dsacq_decomp_kernel=25,
        dsacq_level_slope_k=25,
        cross_attn_layers=run["cross_attn_layers"],
        sacq_mem_film_enable=0,
        sacq_mem_film_hidden=64,
        sacq_mem_film_scale=0.25,
        sacq_autoregressive=1,
        sacq_future_causal=0,
        sacq_ar_verify=0,
        sacq_coarse_init="flatten",
        sacq_routing_topk=0,
        sacq_routing_window=0,
        sacq_scheme="default",
        sacq_dual_struct_gate=0,
        sacq_dual_res_fuse=0,
        sacq_dual_detach=1,
        sacq_dual_scaffold_ln=0,
        sacq_pe_mode="default",
        sacq_pe_rel_buckets=32,
        sacq_time_token_enable=0,
        sacq_time_token_hidden=64,
        sacq_time_token_dropout=0.0,
        sacq_time_token_pool="last",
        sacq_time_token_gate_init=0.05,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        sacq_overlap_avg=0,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        sacq_coarse_lr_ratio=1.0,
        sacq_decoder_lr_ratio=1.0,
        sacq_backbone_lr_ratio=0.3,
        sacq_alpha_lr_ratio=1.0,
        bootstrap_flatten_head=0,
        freeze_backbone=0,
        freeze_coarse_head=0,
        dsacq_enable=0,
        dsacq_mem_gate_enable=0,
        dsacq_drift_gate_enable=0,
        dsacq_mem_gate_hidden=32,
        dsacq_mem_gate_mode="stats",
        mem_shift_aug_std=0.15,
        mem_shift_aug_prob=0.5,
        mem_coral_loss_weight=0.05,
        align_ablate_shift=0,
        align_ablate_dual_bank=0,
        align_ablate_token_gate=0,
        align_ablate_shift_drift=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
        enc_in=21,
        dec_in=21,
        c_out=21,
        d_model=128,
        n_heads=16,
        e_layers=3,
        d_layers=1,
        d_ff=256,
        moving_avg=25,
        factor=1,
        distil=False,
        dropout=0.2,
        embed="timeF",
        activation="gelu",
        num_workers=4,
        itr=1,
        train_epochs=30,
        batch_size=128,
        patience=12,
        learning_rate=1e-4,
        des=run["des"],
        short_setting=1,
        loss="logcosh_scaled",
        logcosh_beta=run["logcosh_beta"],
        logcosh_beta_mode="auto_mad",
        drift_stat_loss_weight=0.0,
        sacq_gate_loss_weight=0.0,
        sacq_corr_sign_loss_weight=0.0,
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=0.5,
        sacq_alpha_granularity="patch",
        sacq_alpha_softplus=0,
        sacq_shift_boost_loss_weight=0.0,
        sacq_grad_firewall_enable=0,
        sacq_grad_firewall_hidden=64,
        sacq_grad_firewall_temp=1.0,
        sacq_grad_firewall_mode="learnable",
        sacq_grad_firewall_sup_weight=0.0,
        sacq_grad_firewall_sparse_weight=0.0,
        lradj="TST",
        pct_start=0.2,
        use_amp=False,
        use_gpu=use_gpu,
        gpu=gpu,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
    )


def _flat_to_btc(exp: Exp_Main, flat: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
    flat_btc = exp._to_btc(flat, batch_y)
    if flat_btc is None:
        if flat.shape[1] == batch_y.shape[2]:
            flat_btc = flat.permute(0, 2, 1)
        else:
            flat_btc = flat
    return flat_btc


def eval_coarse_only(exp: Exp_Main, pred_len: int) -> dict:
    _, test_loader = exp._get_data(flag="test")
    preds, trues, inputx = [], [], []
    f_dim = -1 if exp.args.features == "MS" else 0

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            _, aux = exp._split_model_output(exp._forward(batch_x))
            flat = aux.get("sacq_flat_seq") if aux else None
            if flat is None:
                raise RuntimeError("Model did not return sacq_flat_seq in aux dict.")

            flat_btc = _flat_to_btc(exp, flat, batch_y)
            pred = flat_btc[:, -pred_len:, f_dim:]
            true = batch_y[:, -pred_len:, f_dim:]
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            inputx.append(batch_x.detach().cpu().numpy())

    preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
    inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(
        preds, trues, inputx=inputx
    )
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rse": float(rse),
        "rmse": float(rmse),
        "corr": float(corr),
    }


def _resolve_ckpt(ckpt_dir: str, model_id: str) -> str:
    for name in (f"{model_id}_checkpoint.pth", f"{model_id}.pth"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"checkpoint not found for {model_id} in {ckpt_dir} "
        f"(tried {model_id}_checkpoint.pth and {model_id}.pth)"
    )


def eval_one(pred_len: int, ckpt_dir: str, gpu: int, use_gpu: bool = True) -> dict:
    run = WEATHER_RUNS[pred_len]
    ckpt = _resolve_ckpt(ckpt_dir, run["model_id"])

    ns = make_weather_args(pred_len, gpu=gpu, use_gpu=use_gpu)
    exp = Exp_Main(ns)
    map_loc = exp.device if use_gpu else "cpu"
    exp.model.load_state_dict(torch.load(ckpt, map_location=map_loc))
    print(f"[pl={pred_len}] loaded {ckpt}", flush=True)

    metrics = eval_coarse_only(exp, pred_len)
    metrics["pred_len"] = pred_len
    metrics["model_id"] = run["model_id"]
    metrics["setting"] = run["setting"]
    metrics["checkpoint"] = ckpt
    metrics["full_sar_mse"] = run["full_mse"]
    metrics["full_sar_mae"] = run["full_mae"]
    metrics["mse_delta_pct"] = (metrics["mse"] - run["full_mse"]) / run["full_mse"] * 100.0
    metrics["mae_delta_pct"] = (metrics["mae"] - run["full_mae"]) / run["full_mae"] * 100.0
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Weather coarse-only eval (sacq_flat_seq) for 4 pred_lens."
    )
    parser.add_argument(
        "--pred_lens",
        type=int,
        nargs="+",
        default=PRED_LENS,
        choices=PRED_LENS,
        help="prediction lengths to evaluate (default: all four)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="directory containing Ws512p*_checkpoint.pth files",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU inference (use when CUDA driver unavailable)",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="optional path to write all results as JSON",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    results = []
    print("=== Weather coarse-only (sacq_flat_seq) test metrics ===\n")
    print(f"{'pl':>4}  {'coarse_mse':>10}  {'coarse_mae':>10}  {'full_mse':>10}  {'delta%':>8}")
    print("-" * 50)

    for pl in args.pred_lens:
        m = eval_one(pl, args.ckpt_dir, args.gpu, use_gpu=not args.cpu)
        results.append(m)
        print(
            f"{pl:4d}  {m['mse']:10.6f}  {m['mae']:10.6f}  "
            f"{m['full_sar_mse']:10.6f}  {m['mse_delta_pct']:+7.2f}%"
        )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
