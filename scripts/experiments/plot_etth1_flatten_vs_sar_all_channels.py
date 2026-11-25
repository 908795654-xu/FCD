"""ETTh1 sl512: all-channel curves — Flatten+MSE vs SAR+logcosh_scaled (4 pred_lens)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main

PRED_LENS = [96, 192, 336, 720]

FLATTEN_SETTINGS = {
    96: "ETTh1_512_96_PatchTST_ETTh1_sl512_pl96_Exp_0",
    192: "ETTh1_512_192_PatchTST_ETTh1_sl512_pl192_Exp_0",
    336: "ETTh1_512_336_PatchTST_ETTh1_sl512_pl336_Exp_0",
    720: "ETTh1_512_720_PatchTST_ETTh1_sl512_pl720_Exp_0",
}

SAR_SETTINGS = {
    96: "H1s512p96SARslc_PatchTST_ETTh1_sl512_pl96_hSARslc_0",
    192: "H1s512p192SARslc_PatchTST_ETTh1_sl512_pl192_hSARslc_0",
    336: "H1s512p336SARslc_PatchTST_ETTh1_sl512_pl336_hSARslc_0",
    720: "H1s512p720SARslc_PatchTST_ETTh1_sl512_pl720_hSARslc_0",
}


def _etth1_base(pred_len: int, **overrides) -> Namespace:
    base = dict(
        random_seed=2021,
        is_training=0,
        model="PatchTST",
        data="ETTh1",
        root_path="./dataset/",
        data_path="ETTh1.csv",
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
        pred_len=pred_len,
        fc_dropout=0.3,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        n_heads=4,
        e_layers=3,
        d_layers=1,
        d_ff=128,
        dropout=0.3,
        embed="timeF",
        factor=1,
        distil=True,
        individual=0,
        decomposition=0,
        kernel_size=25,
        num_workers=0,
        itr=1,
        train_epochs=30,
        batch_size=128,
        patience=12,
        learning_rate=1e-4,
        des="Exp",
        short_setting=1,
        lradj="TST",
        pct_start=0.2,
        use_amp=False,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
        loss="mse",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
        cross_attn_layers=2,
        sacq_autoregressive=0,
        sacq_scheme="default",
        sacq_alpha_mode="learnable",
        dsacq_enable=0,
    )
    base.update(overrides)
    return Namespace(**base)


def make_flatten_args(pred_len: int) -> Namespace:
    pl = pred_len
    return _etth1_base(
        pl,
        model_id=f"ETTh1_512_{pl}",
        head_type="flatten",
        loss="mse",
    )


def make_sar_args(pred_len: int) -> Namespace:
    pl = pred_len
    return _etth1_base(
        pl,
        model_id=f"H1s512p{pl}SARslc",
        head_type="sacq_cross_attention",
        sacq_autoregressive=1,
        loss="logcosh_scaled",
        des="hSARslc",
    )


def _set_device(ns: Namespace, gpu: int, force_cpu: bool) -> None:
    if force_cpu or not torch.cuda.is_available():
        ns.use_gpu = False
        ns.gpu = 0
    else:
        ns.use_gpu = True
        ns.gpu = gpu


def _to_btc(exp, tensor, batch_y):
    out = exp._to_btc(tensor, batch_y)
    if out is not None:
        return out
    if tensor.shape[1] == batch_y.shape[2]:
        return tensor.permute(0, 2, 1)
    return tensor


def collect_one_sample(
    exp: Exp_Main,
    setting: str,
    sample_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"loaded {ckpt} on {exp.device}", flush=True)

    bs = exp.args.batch_size
    target_batch = sample_idx // bs
    offset = sample_idx % bs

    _, test_loader = exp._get_data(flag="test")
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if bi < target_batch:
                continue
            if bi > target_batch:
                raise IndexError(
                    f"sample_idx={sample_idx} past test loader (batch {bi} > {target_batch})"
                )
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            out, _ = exp._split_model_output(exp._forward(batch_x))
            pred_btc = _to_btc(exp, out, batch_y)
            inp = batch_x[offset].detach().cpu().numpy()
            pred = pred_btc[offset, -pl:, f_dim:].detach().cpu().numpy()
            true = batch_y[offset, -pl:, f_dim:].detach().cpu().numpy()
            return inp, pred, true
    raise IndexError(f"sample_idx={sample_idx} not found in test loader")


def collect_all_preds(exp: Exp_Main, setting: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))

    _, test_loader = exp._get_data(flag="test")
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len
    preds, trues = [], []

    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in test_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            out, _ = exp._split_model_output(exp._forward(batch_x))
            pred_btc = _to_btc(exp, out, batch_y)
            preds.append(pred_btc[:, -pl:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -pl:, f_dim:].detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def pick_sample_index(flat_p, sar_p, trues, channel: int, mode: str) -> int:
    err_flat = np.mean((flat_p[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    err_sar = np.mean((sar_p[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    gain = err_flat - err_sar
    if mode == "max_gain":
        return int(np.argmax(gain))
    if mode == "max_flat_err":
        return int(np.argmax(err_flat))
    if mode == "max_sar_err":
        return int(np.argmax(err_sar))
    return 0


def load_test_mse(setting: str) -> float | None:
    p = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f).get("mse", float("nan")))
    return None


def _channel_names(n_ch: int) -> list[str]:
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "ETTh1.csv")
    if os.path.isfile(csv_path):
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        cols = [c for c in cols if c != "date"]
        if len(cols) >= n_ch:
            return cols[:n_ch]
    return [f"ch{i}" for i in range(n_ch)]


def plot_all_channels(
    inp: np.ndarray,
    true: np.ndarray,
    flat_p: np.ndarray,
    sar_p: np.ndarray,
    names: list[str],
    out_path: str,
    title: str,
    hist_len: int,
    zoom_len: int,
):
    n_ch = true.shape[-1]
    ncols = 3
    nrows = int(np.ceil(n_ch / ncols))
    seq_len = inp.shape[0]
    h0 = max(0, seq_len - hist_len)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 2.5), sharex=True)
    axes = np.atleast_2d(axes)

    for c in range(n_ch):
        ax = axes[c // ncols, c % ncols]
        y_true = true[:zoom_len, c]
        y_flat = flat_p[:zoom_len, c]
        y_sar = sar_p[:zoom_len, c]
        t_hist = np.arange(h0, seq_len)
        t_fut = np.arange(seq_len, seq_len + zoom_len)

        ax.plot(t_hist, inp[h0:, c], color="#bdbdbd", lw=0.8, alpha=0.8)
        ax.plot(t_fut, y_true, color="#1f77b4", lw=1.4, label="truth")
        ax.plot(t_fut, y_flat, color="#ff7f0e", lw=1.0, ls="--", label="Flatten+MSE")
        ax.plot(t_fut, y_sar, color="#d62728", lw=1.0, ls="-", label="SAR+logcosh")
        ax.axvline(seq_len - 0.5, color="k", ls=":", lw=0.6, alpha=0.4)
        loc_flat = float(np.mean((y_flat - y_true) ** 2))
        loc_sar = float(np.mean((y_sar - y_true) ** 2))
        ax.set_title(f"{names[c]}  Flat={loc_flat:.4f} SAR={loc_sar:.4f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for c in range(n_ch, nrows * ncols):
        axes[c // ncols, c % ncols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11, y=1.002)
    fig.supxlabel("time step", fontsize=10)
    fig.supylabel("scaled value", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_all_channels_stacked(
    inp: np.ndarray,
    true: np.ndarray,
    flat_p: np.ndarray,
    sar_p: np.ndarray,
    names: list[str],
    out_path: str,
    title: str,
    hist_len: int,
    zoom_len: int,
    panel_h: float = 2.2,
    fig_w: float = 18.0,
    dpi: int = 150,
):
    """One wide panel per channel, stacked vertically."""
    n_ch = true.shape[-1]
    seq_len = inp.shape[0]
    h0 = max(0, seq_len - hist_len)

    fig, axes = plt.subplots(
        n_ch,
        1,
        figsize=(fig_w, panel_h * n_ch),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for c in range(n_ch):
        ax = axes[c]
        y_true = true[:zoom_len, c]
        y_flat = flat_p[:zoom_len, c]
        y_sar = sar_p[:zoom_len, c]
        t_hist = np.arange(h0, seq_len)
        t_fut = np.arange(seq_len, seq_len + zoom_len)

        ax.plot(t_hist, inp[h0:, c], color="#bdbdbd", lw=1.0, alpha=0.85, label="history")
        ax.plot(t_fut, y_true, color="#1f77b4", lw=1.8, label="truth")
        ax.plot(t_fut, y_flat, color="#ff7f0e", lw=1.4, ls="--", label="Flatten+MSE")
        ax.plot(t_fut, y_sar, color="#d62728", lw=1.5, ls="-", label="SAR+logcosh")
        ax.axvline(seq_len - 0.5, color="k", ls=":", lw=0.7, alpha=0.45)
        loc_flat = float(np.mean((y_flat - y_true) ** 2))
        loc_sar = float(np.mean((y_sar - y_true) ** 2))
        ax.set_ylabel(names[c], fontsize=10, rotation=0, ha="right", va="center", labelpad=28)
        ax.text(
            0.995,
            0.92,
            f"Flat MSE={loc_flat:.4f}  SAR MSE={loc_sar:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.22)
        if c < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("time step", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(title, fontsize=12, y=1.008)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.04, hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def replot_from_npz(npz_path: str, out_path: str, hist_len: int, zoom_len: int, title: str | None = None):
    d = np.load(npz_path, allow_pickle=True)
    inp = d["inputs"]
    true_m = d["true"]
    flat_p = d["flatten"]
    sar_p = d["sar"]
    si = int(d["sample_idx"])
    pl = int(d["pred_len"]) if "pred_len" in d else true_m.shape[0]
    names = list(d["channel_names"]) if "channel_names" in d else _channel_names(true_m.shape[-1])
    if zoom_len <= 0:
        zoom_len = pl
    if title is None:
        title = f"ETTh1 sl512 pl={pl} | sample={si} | zoom={zoom_len}"
    plot_all_channels_stacked(inp, true_m, flat_p, sar_p, names, out_path, title, hist_len, zoom_len)


def load_summary_mse(csv_path: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    if not os.path.isfile(csv_path):
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pl = int(row.get("pred_len", 0))
            mse = row.get("mse")
            if mse:
                out[pl] = float(mse)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=-1, help="-1: auto per pred_len")
    parser.add_argument("--pick", type=str, default="max_gain", choices=["max_gain", "max_flat_err", "max_sar_err", "first"])
    parser.add_argument("--channel", type=int, default=0, help="channel for auto sample pick")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth1_flatten_vs_sar_curves",
    )
    parser.add_argument("--hist_len", type=int, default=128)
    parser.add_argument(
        "--zoom_len",
        type=int,
        default=0,
        help="0 = use full pred_len per figure",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="grid",
        choices=["grid", "stacked"],
        help="grid: 3-column mosaic; stacked: one wide panel per channel vertically",
    )
    parser.add_argument(
        "--pred_lens",
        type=str,
        default="",
        help="comma-separated pred_lens to plot, e.g. 720; empty = all",
    )
    parser.add_argument(
        "--replot_npz",
        type=str,
        default="",
        help="replot from existing .npz only (skip inference)",
    )
    args = parser.parse_args()

    if args.replot_npz:
        os.chdir(PROJECT_ROOT)
        npz_path = args.replot_npz
        if not os.path.isabs(npz_path):
            npz_path = os.path.join(PROJECT_ROOT, npz_path)
        out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(npz_path))[0]
        suffix = "_stacked" if args.layout == "stacked" else ""
        d_probe = np.load(npz_path, allow_pickle=True)
        pl_probe = int(d_probe["pred_len"]) if "pred_len" in d_probe else int(d_probe["true"].shape[0])
        zoom_use = args.zoom_len if args.zoom_len > 0 else pl_probe
        zoom_tag = f"_zoom{zoom_use}"
        out_path = os.path.join(out_dir, f"{base}{suffix}{zoom_tag}.png")
        replot_from_npz(npz_path, out_path, args.hist_len, zoom_use)
        print(f"saved -> {out_path}", flush=True)
        return

    pred_lens = PRED_LENS
    if args.pred_lens.strip():
        pred_lens = [int(x) for x in args.pred_lens.split(",") if x.strip()]

    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    plot_fn = plot_all_channels_stacked if args.layout == "stacked" else plot_all_channels

    for pred_len in pred_lens:
        flat_setting = FLATTEN_SETTINGS[pred_len]
        sar_setting = SAR_SETTINGS[pred_len]
        zoom_len = args.zoom_len if args.zoom_len > 0 else pred_len

        print(f"\n=== ETTh1 pl={pred_len} ===", flush=True)

        if args.sample_idx < 0:
            ns_f = make_flatten_args(pred_len)
            _set_device(ns_f, args.gpu, args.cpu)
            exp_f = Exp_Main(ns_f)
            flat_all, trues_all = collect_all_preds(exp_f, flat_setting)
            del exp_f, ns_f

            ns_s = make_sar_args(pred_len)
            _set_device(ns_s, args.gpu, args.cpu)
            exp_s = Exp_Main(ns_s)
            sar_all, trues_s = collect_all_preds(exp_s, sar_setting)
            del exp_s, ns_s

            if not np.allclose(trues_all, trues_s):
                print("WARNING: ground truth mismatch", flush=True)
            ch = min(args.channel, trues_all.shape[-1] - 1)
            si = pick_sample_index(flat_all, sar_all, trues_all, ch, args.pick)
            note = f"auto sample={si} ({args.pick}, ch={ch})"
            del flat_all, sar_all, trues_all
        else:
            si = args.sample_idx
            note = f"sample={si}"

        ns_flat = make_flatten_args(pred_len)
        _set_device(ns_flat, args.gpu, args.cpu)
        inp, flat_p, true_m = collect_one_sample(Exp_Main(ns_flat), flat_setting, si)
        del ns_flat

        ns_sar = make_sar_args(pred_len)
        _set_device(ns_sar, args.gpu, args.cpu)
        _, sar_p, true_s = collect_one_sample(Exp_Main(ns_sar), sar_setting, si)
        del ns_sar

        if not np.allclose(true_m, true_s):
            print("WARNING: ground truth mismatch across loaders", flush=True)

        names = _channel_names(true_m.shape[-1])
        gf = load_test_mse(flat_setting)
        gs = load_test_mse(sar_setting)
        title = (
            f"ETTh1 sl512 pl={pred_len} all channels | {note} | zoom={zoom_len}\n"
            f"test MSE: Flatten+MSE={gf:.4f} | SAR+logcosh={gs:.4f}"
            if gf is not None and gs is not None
            else f"ETTh1 sl512 pl={pred_len} all channels | {note}"
        )

        base = f"etth1_pl{pred_len}_sample{si}_allch"
        stacked_suffix = "_stacked" if args.layout == "stacked" else ""
        out_path = os.path.join(out_dir, f"{base}{stacked_suffix}_zoom{zoom_len}.png")
        plot_fn(inp, true_m, flat_p, sar_p, names, out_path, title, args.hist_len, zoom_len)
        npz_path = os.path.join(out_dir, f"{base}.npz")
        np.savez(
            npz_path,
            inputs=inp,
            true=true_m,
            flatten=flat_p,
            sar=sar_p,
            sample_idx=si,
            pred_len=pred_len,
            channel_names=np.array(names, dtype=object),
        )
        print(f"saved -> {out_path}", flush=True)
        print(f"npz  -> {npz_path}", flush=True)


if __name__ == "__main__":
    main()
