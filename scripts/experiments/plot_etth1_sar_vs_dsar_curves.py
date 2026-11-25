"""ETTh1 pl96 (etc.): SAR+logcosh vs DSAR+logcosh — all-channel stacked curves on one figure."""
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main
from scripts.experiments.plot_etth1_flatten_vs_sar_all_channels import (
    SAR_SETTINGS,
    _channel_names,
    _etth1_base,
    _set_device,
    collect_all_preds,
    collect_one_sample,
    make_sar_args,
)

SAR_SETTINGS_PL = {96: SAR_SETTINGS[96]}

DSAR_SETTINGS = {
    96: "H1s512p96DSARslc_PatchTST_ETTh1_sl512_pl96_hDSARslc_0",
    192: "H1s512p192DSARslc_PatchTST_ETTh1_sl512_pl192_hDSARslc_0",
    336: "H1s512p336DSARslc_PatchTST_ETTh1_sl512_pl336_hDSARslc_0",
    720: "H1s512p720DSARslc_PatchTST_ETTh1_sl512_pl720_hDSARslc_0",
}


def make_dsar_args(pred_len: int):
    pl = pred_len
    return _etth1_base(
        pl,
        model_id=f"H1s512p{pl}DSARslc",
        head_type="dsacq_cross_attention",
        sacq_autoregressive=1,
        dsacq_enable=1,
        dsacq_mem_gate_enable=1,
        dsacq_drift_gate_enable=1,
        dsacq_coarse_split=1,
        loss="logcosh_scaled",
        des="hDSARslc",
    )


def pick_sample(sar_p, dsar_p, trues, ch: int, mode: str) -> int:
    err_sar = np.mean((sar_p[:, :, ch] - trues[:, :, ch]) ** 2, axis=1)
    err_dsar = np.mean((dsar_p[:, :, ch] - trues[:, :, ch]) ** 2, axis=1)
    gain = err_dsar - err_sar
    if mode == "max_gain":
        return int(np.argmax(gain))
    if mode == "max_sar_err":
        return int(np.argmax(err_sar))
    if mode == "max_dsar_err":
        return int(np.argmax(err_dsar))
    return 0


def load_test_mse(setting: str) -> float | None:
    p = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f)["mse"])
    return None


def plot_sar_vs_dsar_stacked(
    inp,
    true,
    sar_p,
    dsar_p,
    names,
    out_path,
    title,
    hist_len,
    zoom_len,
    fig_w=18.0,
    panel_h=2.2,
    dpi=150,
):
    n_ch = true.shape[-1]
    seq_len = inp.shape[0]
    h0 = max(0, seq_len - hist_len)
    fig, axes = plt.subplots(n_ch, 1, figsize=(fig_w, panel_h * n_ch), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for c in range(n_ch):
        ax = axes[c]
        y_true = true[:zoom_len, c]
        y_sar = sar_p[:zoom_len, c]
        y_dsar = dsar_p[:zoom_len, c]
        t_hist = np.arange(h0, seq_len)
        t_fut = np.arange(seq_len, seq_len + zoom_len)

        ax.plot(t_hist, inp[h0:, c], color="#bdbdbd", lw=1.0, alpha=0.85, label="history")
        ax.plot(t_fut, y_true, color="#1f77b4", lw=1.8, label="truth")
        ax.plot(t_fut, y_sar, color="#2ca02c", lw=1.4, ls="-", label="SAR+logcosh")
        ax.plot(t_fut, y_dsar, color="#d62728", lw=1.4, ls="--", label="DSAR+logcosh")
        ax.axvline(seq_len - 0.5, color="k", ls=":", lw=0.7, alpha=0.45)
        loc_sar = float(np.mean((y_sar - y_true) ** 2))
        loc_dsar = float(np.mean((y_dsar - y_true) ** 2))
        ax.set_ylabel(names[c], fontsize=10, rotation=0, ha="right", va="center", labelpad=28)
        ax.text(
            0.995,
            0.92,
            f"SAR={loc_sar:.4f}  DSAR={loc_dsar:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )
        ax.grid(True, alpha=0.22)
        if c < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("time step", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=12, y=1.008)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.04, hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument(
        "--pick",
        type=str,
        default="max_gain",
        choices=["max_gain", "max_sar_err", "max_dsar_err", "first"],
    )
    parser.add_argument("--channel", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth1_sar_vs_dsar_curves",
    )
    parser.add_argument("--hist_len", type=int, default=128)
    parser.add_argument("--zoom_len", type=int, default=0)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    pl = args.pred_len
    sar_setting = SAR_SETTINGS[pl]
    dsar_setting = DSAR_SETTINGS[pl]
    zoom_len = args.zoom_len if args.zoom_len > 0 else pl
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.sample_idx < 0:
        ns_s = make_sar_args(pl)
        _set_device(ns_s, args.gpu, args.cpu)
        sar_all, trues = collect_all_preds(Exp_Main(ns_s), sar_setting)
        del ns_s

        ns_d = make_dsar_args(pl)
        _set_device(ns_d, args.gpu, args.cpu)
        dsar_all, trues_d = collect_all_preds(Exp_Main(ns_d), dsar_setting)
        del ns_d
        if not np.allclose(trues, trues_d):
            print("WARNING: ground truth mismatch", flush=True)
        ch = min(args.channel, trues.shape[-1] - 1)
        si = pick_sample(sar_all, dsar_all, trues, ch, args.pick)
        ch_names = _channel_names(trues.shape[-1])
        note = f"auto sample={si} ({args.pick}, ch={ch_names[ch]})"
    else:
        si = args.sample_idx
        note = f"sample={si}"

    ns_sar = make_sar_args(pl)
    _set_device(ns_sar, args.gpu, args.cpu)
    inp, sar_p, true_m = collect_one_sample(Exp_Main(ns_sar), sar_setting, si)
    del ns_sar

    ns_dsar = make_dsar_args(pl)
    _set_device(ns_dsar, args.gpu, args.cpu)
    _, dsar_p, true_d = collect_one_sample(Exp_Main(ns_dsar), dsar_setting, si)
    del ns_dsar

    if not np.allclose(true_m, true_d):
        print("WARNING: ground truth mismatch across loaders", flush=True)

    names = _channel_names(true_m.shape[-1])
    gs = load_test_mse(sar_setting)
    gd = load_test_mse(dsar_setting)
    title = f"ETTh1 pl={pl} SAR vs DSAR | {note} | zoom={zoom_len}"
    if gs is not None and gd is not None:
        title += f"\ntest MSE: SAR={gs:.4f} | DSAR={gd:.4f}"

    base = f"etth1_pl{pl}_sample{si}_sar_vs_dsar"
    out_path = os.path.join(out_dir, f"{base}_stacked_zoom{zoom_len}.png")
    plot_sar_vs_dsar_stacked(
        inp, true_m, sar_p, dsar_p, names, out_path, title, args.hist_len, zoom_len
    )
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        inputs=inp,
        true=true_m,
        sar=sar_p,
        dsar=dsar_p,
        sample_idx=si,
        pred_len=pl,
        channel_names=np.array(names, dtype=object),
    )
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
