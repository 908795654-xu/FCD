"""Plot ETTh1 DSAR (DSACQ+AR+logcosh_scaled) test-set prediction curves."""
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
    _channel_names,
    _etth1_base,
    _set_device,
    _to_btc,
    collect_all_preds,
    collect_one_sample,
)

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


def pick_sample(preds, trues, ch: int, mode: str) -> int:
    err = np.mean((preds[:, :, ch] - trues[:, :, ch]) ** 2, axis=1)
    if mode == "max_err":
        return int(np.argmax(err))
    if mode == "median_err":
        return int(np.argsort(err)[len(err) // 2])
    return 0


def plot_stacked(
    inp, true, pred, names, out_path, title, hist_len, zoom_len, fig_w=18.0, panel_h=2.2, dpi=150
):
    n_ch = true.shape[-1]
    seq_len = inp.shape[0]
    h0 = max(0, seq_len - hist_len)
    fig, axes = plt.subplots(n_ch, 1, figsize=(fig_w, panel_h * n_ch), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for c in range(n_ch):
        ax = axes[c]
        y_true = true[:zoom_len, c]
        y_pred = pred[:zoom_len, c]
        t_hist = np.arange(h0, seq_len)
        t_fut = np.arange(seq_len, seq_len + zoom_len)
        ax.plot(t_hist, inp[h0:, c], color="#bdbdbd", lw=1.0, alpha=0.85, label="history")
        ax.plot(t_fut, y_true, color="#1f77b4", lw=1.8, label="truth")
        ax.plot(t_fut, y_pred, color="#d62728", lw=1.5, ls="--", label="DSAR pred")
        ax.axvline(seq_len - 0.5, color="k", ls=":", lw=0.7, alpha=0.45)
        loc = float(np.mean((y_pred - y_true) ** 2))
        ax.set_ylabel(names[c], fontsize=10, rotation=0, ha="right", va="center", labelpad=28)
        ax.text(
            0.995, 0.92, f"MSE={loc:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )
        ax.grid(True, alpha=0.22)
        if c < n_ch - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("time step", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle(title, fontsize=12, y=1.008)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.04, hspace=0.12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_test_mse(setting: str) -> float | None:
    p = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f)["mse"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--setting", type=str, default="")
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument("--pick", type=str, default="max_err", choices=["max_err", "median_err", "first"])
    parser.add_argument("--channel", type=int, default=6, help="channel for auto sample (6=OT)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", type=str, default="logs/etth1_dsar_sl512_logcosh_scaled/curves")
    parser.add_argument("--hist_len", type=int, default=128)
    parser.add_argument("--zoom_len", type=int, default=0)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    pl = args.pred_len
    setting = args.setting or DSAR_SETTINGS[pl]
    zoom_len = args.zoom_len if args.zoom_len > 0 else pl
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ns = make_dsar_args(pl)
    _set_device(ns, args.gpu, args.cpu)
    exp = Exp_Main(ns)

    if args.sample_idx < 0:
        preds, trues = collect_all_preds(exp, setting)
        ch = min(args.channel, trues.shape[-1] - 1)
        si = pick_sample(preds, trues, ch, args.pick)
        ch_names = _channel_names(trues.shape[-1])
        note = f"auto sample={si} ({args.pick}, ch={ch_names[ch]})"
    else:
        si = args.sample_idx
        note = f"sample={si}"

    inp, pred, true = collect_one_sample(exp, setting, si)
    names = _channel_names(true.shape[-1])
    gm = load_test_mse(setting)
    title = f"ETTh1 DSAR pl={pl} | {note} | zoom={zoom_len}"
    if gm is not None:
        title += f"\ntest MSE={gm:.4f}"

    base = f"etth1_dsar_pl{pl}_sample{si}_allch"
    out_path = os.path.join(out_dir, f"{base}_stacked_zoom{zoom_len}.png")
    plot_stacked(inp, true, pred, names, out_path, title, args.hist_len, zoom_len)
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        inputs=inp, true=true, pred=pred, sample_idx=si, pred_len=pl,
        channel_names=np.array(names, dtype=object), setting=setting,
    )
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
