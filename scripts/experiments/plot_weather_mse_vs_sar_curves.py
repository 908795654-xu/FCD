"""Plot Weather pl720 test curves: MSE Flatten vs Flatten+logcosh vs SAR."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main

MSE_SETTING = (
    "weather_512_720_PatchTST_weather_ftM_sl512_ll48_pl720_"
    "dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
)
LC_SETTING = (
    "weather_512_720_lc_PatchTST_weather_ftM_sl512_ll48_pl720_"
    "dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
)
SAR_SETTING = (
    "weather_512_720_SAR_PatchTST_weather_ftM_sl512_ll48_pl720_"
    "dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
)


def _weather_base(**overrides) -> Namespace:
    base = dict(
        random_seed=2021,
        is_training=0,
        model="PatchTST",
        data="weather",
        root_path="./dataset/",
        data_path="weather.csv",
        scaler_data_path="",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        checkpoint_setting="",
        init_checkpoint_setting="",
        skip_visual=1,
        eval_tag="curve_plot",
        seq_len=512,
        label_len=48,
        pred_len=720,
        fc_dropout=0.2,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
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
        dsacq_enable=1,
        dsacq_mem_gate_enable=1,
        dsacq_drift_gate_enable=1,
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
        num_workers=0,
        itr=1,
        train_epochs=30,
        batch_size=128,
        patience=12,
        learning_rate=1e-4,
        des="Exp",
        short_setting=0,
        drift_stat_loss_weight=0.0,
        sacq_gate_loss_weight=0.0,
        sacq_corr_sign_loss_weight=0.0,
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=0.5,
        sacq_alpha_granularity="patch",
        sacq_alpha_softplus=0,
        sacq_shift_boost_loss_weight=0.0,
        lradj="type3",
        pct_start=0.3,
        use_amp=False,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
        loss="mse",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
    )
    base.update(overrides)
    return Namespace(**base)


def make_mse_args() -> Namespace:
    return _weather_base(
        model_id="weather_512_720",
        head_type="flatten",
        cross_attn_layers=2,
        sacq_autoregressive=0,
        loss="mse",
    )


def make_lc_args() -> Namespace:
    return _weather_base(
        model_id="weather_512_720_lc",
        head_type="flatten",
        cross_attn_layers=2,
        sacq_autoregressive=0,
        loss="logcosh_scaled",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
    )


def make_sar_args() -> Namespace:
    return _weather_base(
        model_id="weather_512_720_SAR",
        head_type="sacq_cross_attention",
        cross_attn_layers=2,
        sacq_autoregressive=1,
        loss="logcosh_scaled",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
    )


def _to_btc(exp, tensor, batch_y):
    out = exp._to_btc(tensor, batch_y)
    if out is not None:
        return out
    if tensor.shape[1] == batch_y.shape[2]:
        return tensor.permute(0, 2, 1)
    return tensor


def collect_preds(exp: Exp_Main, setting: str, max_batches: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"loaded {ckpt} on {exp.device}", flush=True)

    _, test_loader = exp._get_data(flag="test")
    preds, trues, inputs = [], [], []
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            out, _ = exp._split_model_output(exp._forward(batch_x))
            pred_btc = _to_btc(exp, out, batch_y)
            preds.append(pred_btc[:, -pl:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -pl:, f_dim:].detach().cpu().numpy())
            inputs.append(batch_x.detach().cpu().numpy())

    return (
        np.concatenate(inputs, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
    )


def pick_sample_index(mse_preds, sar_preds, trues, channel: int, mode: str) -> int:
    err_mse = np.mean((mse_preds[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    err_sar = np.mean((sar_preds[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    gain = err_mse - err_sar
    if mode == "max_gain":
        return int(np.argmax(gain))
    if mode == "max_mse_err":
        return int(np.argmax(err_mse))
    if mode == "max_sar_err":
        return int(np.argmax(err_sar))
    return 0


def plot_compare(
    inp_hist,
    true_fut,
    mse_fut,
    lc_fut,
    sar_fut,
    out_path: str,
    title: str,
    global_mse: dict[str, float | None],
    zoom_len: int | None = None,
):
    seq_len = len(inp_hist)
    pred_len = len(true_fut)
    if zoom_len is not None and zoom_len < pred_len:
        true_fut = true_fut[:zoom_len]
        mse_fut = mse_fut[:zoom_len]
        lc_fut = lc_fut[:zoom_len]
        sar_fut = sar_fut[:zoom_len]
        pred_len = zoom_len
        title = title + f" (first {zoom_len} steps)"

    t_in = np.arange(seq_len)
    t_fut = np.arange(seq_len, seq_len + pred_len)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(t_in, inp_hist, color="#9e9e9e", linewidth=1.0, alpha=0.75, label="history")
    ax.plot(t_fut, true_fut, color="#1f77b4", linewidth=2.0, label="ground truth")
    ax.plot(
        t_fut,
        mse_fut,
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
        label=f"MSE Flatten (local={np.mean((mse_fut - true_fut) ** 2):.4f})",
    )
    ax.plot(
        t_fut,
        lc_fut,
        color="#2ca02c",
        linewidth=1.5,
        linestyle="-.",
        label=f"Flatten+logcosh (local={np.mean((lc_fut - true_fut) ** 2):.4f})",
    )
    ax.plot(
        t_fut,
        sar_fut,
        color="#d62728",
        linewidth=1.6,
        linestyle="-",
        label=f"SAR+logcosh (local={np.mean((sar_fut - true_fut) ** 2):.4f})",
    )
    ax.axvline(seq_len - 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("time step")
    ax.set_ylabel("value (scaled)")
    parts = []
    for key, label in [("mse", "MSE"), ("lc", "LC"), ("sar", "SAR")]:
        v = global_mse.get(key)
        if v is not None:
            parts.append(f"{label}={v:.4f}")
    if parts:
        title = title + "\ntest MSE: " + " | ".join(parts)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_test_mse(setting: str) -> float | None:
    p = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f).get("mse", 0))
    return None


def try_load_npz(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/weather_pl720_mse_vs_sar/curves",
    )
    parser.add_argument("--max_batches", type=int, default=0, help="0 = full test set")
    parser.add_argument("--sample_idx", type=int, default=2422, help="default: same as prior plot")
    parser.add_argument(
        "--pick",
        type=str,
        default="max_gain",
        choices=["max_gain", "max_mse_err", "max_sar_err", "first"],
    )
    parser.add_argument("--channel", type=int, default=20, help="OT index for weather (21 vars)")
    parser.add_argument("--zoom_len", type=int, default=240)
    parser.add_argument(
        "--reuse_npz",
        action="store_true",
        help="reuse mse/sar curves from existing npz when sample_idx matches",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    max_batches = None if args.max_batches <= 0 else args.max_batches
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ch = args.channel

    npz_path = os.path.join(out_dir, f"weather_pl720_sample{args.sample_idx}_ch{ch}.npz")
    reused = False
    if args.reuse_npz and args.sample_idx >= 0:
        cached = try_load_npz(npz_path)
        if cached is not None and "mse" in cached and "sar" in cached:
            print(f"reuse mse/sar from {npz_path}", flush=True)
            inp_ch = cached["inputs"]
            mse_f = cached["mse"]
            sar_f = cached["sar"]
            true_f = cached["true"]
            si = int(cached.get("sample_idx", args.sample_idx))
            note = f"sample={si}, ch={ch} (reused npz)"
            reused = True

    if not reused:
        print("=== MSE Flatten ===", flush=True)
        exp_mse = Exp_Main(make_mse_args())
        inp_m, mse_preds, trues_m = collect_preds(exp_mse, MSE_SETTING, max_batches)
        del exp_mse

        print("=== SAR ===", flush=True)
        exp_sar = Exp_Main(make_sar_args())
        _, sar_preds, trues_s = collect_preds(exp_sar, SAR_SETTING, max_batches)
        del exp_sar

        n = min(len(mse_preds), len(sar_preds))
        mse_preds, sar_preds = mse_preds[:n], sar_preds[:n]
        trues = trues_m[:n]
        ch = min(ch, trues.shape[-1] - 1)
        if args.sample_idx < 0:
            si = pick_sample_index(mse_preds, sar_preds, trues, ch, args.pick)
            note = f"auto sample={si} ({args.pick}, ch={ch})"
        else:
            si = min(args.sample_idx, n - 1)
            note = f"sample={si}, ch={ch}"

        inp_ch = inp_m[si, :, ch]
        true_f = trues[si, :, ch]
        mse_f = mse_preds[si, :, ch]
        sar_f = sar_preds[si, :, ch]
        npz_path = os.path.join(out_dir, f"weather_pl720_sample{si}_ch{ch}.npz")

    print("=== Flatten+logcosh (weather_512_720_lc) ===", flush=True)
    exp_lc = Exp_Main(make_lc_args())
    _, lc_preds, trues_lc = collect_preds(exp_lc, LC_SETTING, max_batches)
    del exp_lc

    if not reused:
        lc_f = lc_preds[si, :, ch]
    else:
        si = int(cached.get("sample_idx", args.sample_idx))
        lc_f = lc_preds[si, :, ch]
        if not np.allclose(true_f, trues_lc[si, :, ch], atol=1e-5):
            true_f = trues_lc[si, :, ch]

    global_mse = {
        "mse": load_test_mse(MSE_SETTING),
        "lc": load_test_mse(LC_SETTING),
        "sar": load_test_mse(SAR_SETTING),
    }
    base = f"weather_pl720_sample{si}_ch{ch}"
    title = f"Weather pl=720 | MSE Flatten vs Flatten+logcosh vs SAR\n({note})"
    full_path = os.path.join(out_dir, f"{base}_full.png")
    plot_compare(inp_ch, true_f, mse_f, lc_f, sar_f, full_path, title, global_mse)

    if args.zoom_len and args.zoom_len > 0:
        zoom_path = os.path.join(out_dir, f"{base}_zoom{args.zoom_len}.png")
        plot_compare(
            inp_ch, true_f, mse_f, lc_f, sar_f, zoom_path, title, global_mse, zoom_len=args.zoom_len
        )
        print(f"saved zoom -> {zoom_path}", flush=True)

    np.savez(
        npz_path,
        inputs=inp_ch,
        mse=mse_f,
        lc=lc_f,
        sar=sar_f,
        true=true_f,
        sample_idx=si,
        channel=ch,
    )
    print(f"saved full -> {full_path}", flush=True)


if __name__ == "__main__":
    main()
