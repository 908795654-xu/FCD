"""Evaluate SACQ checkpoint using only the Flatten coarse branch (sacq_flat_seq)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from argparse import Namespace

from exp.exp_main import Exp_Main
from utils.metrics import metric


def make_args_from_setting(setting: str, gpu: int = 0) -> Namespace:
    """Build args for a known weather SARslc pl720 run."""
  # Defaults match logs/weather_sar_sl512_logcosh_scaled/logs/Ws512p720SARslc.log
    return Namespace(
        random_seed=2021,
        is_training=0,
        model_id=setting.split("_")[0],
        model="PatchTST",
        data="weather",
        root_path="./dataset/",
        data_path="weather.csv",
        scaler_data_path="",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        checkpoint_setting=setting,
        init_checkpoint_setting="",
        skip_visual=1,
        eval_tag="flatten_only",
        seq_len=512,
        label_len=48,
        pred_len=720,
        fc_dropout=0.2,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        head_type="sacq_cross_attention",
        cross_attn_layers=2,
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
        des="wSARslc",
        short_setting=1,
        loss="logcosh_scaled",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
        drift_stat_loss_weight=0.0,
        sacq_gate_loss_weight=0.0,
        sacq_corr_sign_loss_weight=0.0,
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=0.5,
        sacq_alpha_granularity="patch",
        sacq_shift_boost_loss_weight=0.0,
        lradj="TST",
        pct_start=0.2,
        use_amp=False,
        use_gpu=True,
        gpu=gpu,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
    )


def eval_flatten_only(exp: Exp_Main, pred_len: int) -> dict:
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
            flat_btc = exp._to_btc(flat, batch_y)
            if flat_btc is None:
                if flat.shape[1] == batch_y.shape[2]:
                    flat_btc = flat.permute(0, 2, 1)
                else:
                    flat_btc = flat

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
        "mape": float(mape),
        "mspe": float(mspe),
        "corr": float(corr),
        "smape": float(smape),
        "mase": float(mase),
        "ql50": float(ql50),
        "pwmae": float(pwmae),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=str,
        default="Ws512p720SARslc_PatchTST_weather_sl512_pl720_wSARslc_0",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    ns = make_args_from_setting(args.setting, gpu=args.gpu)
    ns.pred_len = args.pred_len

    exp = Exp_Main(ns)
    ckpt = os.path.join(ns.checkpoints, args.setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"loaded: {ckpt}")

    metrics = eval_flatten_only(exp, args.pred_len)
    full_metrics_path = os.path.join("results", args.setting, "metrics.json")
    full_mse = None
    if os.path.isfile(full_metrics_path):
        with open(full_metrics_path) as f:
            full_mse = json.load(f).get("mse")

    print("\n=== Flatten-only (sacq_flat_seq) test metrics ===")
    print(
        f"mse:{metrics['mse']}, mae:{metrics['mae']}, rse:{metrics['rse']}, "
        f"rmse:{metrics['rmse']}, corr:{metrics['corr']}"
    )
    if full_mse is not None:
        delta = (metrics["mse"] - full_mse) / full_mse * 100
        print(f"full SAR mse={full_mse}  |  flatten-only delta={delta:+.2f}%")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
