"""All-channel curve comparison: MSE Flatten vs Flatten+logcosh vs SAR (Weather pl720)."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main
from scripts.experiments.plot_weather_mse_vs_sar_curves import (
    LC_SETTING,
    MSE_SETTING,
    SAR_SETTING,
    _to_btc,
    load_test_mse,
    make_lc_args,
    make_mse_args,
    make_sar_args,
)


def _set_device(ns, gpu: int, force_cpu: bool) -> None:
    if force_cpu or not torch.cuda.is_available():
        ns.use_gpu = False
        ns.gpu = 0
    else:
        ns.use_gpu = True
        ns.gpu = gpu


def _channel_names(n_ch: int) -> list[str]:
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "weather.csv")
    if os.path.isfile(csv_path):
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        cols = [c for c in cols if c != "date"]
        if len(cols) >= n_ch:
            return cols[:n_ch]
    return [f"ch{i}" for i in range(n_ch)]


def collect_one_sample(
    exp: Exp_Main,
    setting: str,
    sample_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (input [seq, C], pred [pl, C], true [pl, C]) for one test index."""
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


def plot_all_channels(
    inp: np.ndarray,
    true: np.ndarray,
    mse_p: np.ndarray,
    lc_p: np.ndarray,
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

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 2.4), sharex=True)
    axes = np.atleast_2d(axes)

    for c in range(n_ch):
        ax = axes[c // ncols, c % ncols]
        y_true = true[:zoom_len, c]
        y_mse = mse_p[:zoom_len, c]
        y_lc = lc_p[:zoom_len, c]
        y_sar = sar_p[:zoom_len, c]
        t_hist = np.arange(h0, seq_len)
        t_fut = np.arange(seq_len, seq_len + zoom_len)

        ax.plot(t_hist, inp[h0:, c], color="#bdbdbd", lw=0.8, alpha=0.8)
        ax.plot(t_fut, y_true, color="#1f77b4", lw=1.4, label="truth")
        ax.plot(t_fut, y_mse, color="#ff7f0e", lw=1.0, ls="--", label="MSE")
        ax.plot(t_fut, y_lc, color="#2ca02c", lw=1.0, ls="-.", label="LC")
        ax.plot(t_fut, y_sar, color="#d62728", lw=1.0, ls="-", label="SAR")
        ax.axvline(seq_len - 0.5, color="k", ls=":", lw=0.6, alpha=0.4)
        loc_mse = float(np.mean((y_mse - y_true) ** 2))
        loc_sar = float(np.mean((y_sar - y_true) ** 2))
        ax.set_title(f"{names[c]}  MSE={loc_mse:.4f} SAR={loc_sar:.4f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for c in range(n_ch, nrows * ncols):
        axes[c // ncols, c % ncols].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11, y=1.002)
    fig.supxlabel("time step", fontsize=10)
    fig.supylabel("scaled value", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_idx", type=int, default=2422)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/weather_pl720_mse_vs_sar/curves",
    )
    parser.add_argument("--hist_len", type=int, default=128, help="history tail shown per panel")
    parser.add_argument("--zoom_len", type=int, default=240, help="prediction steps per panel")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    si = args.sample_idx
    print(f"=== sample {si} ===", flush=True)

    ns_mse = make_mse_args()
    _set_device(ns_mse, args.gpu, args.cpu)
    inp, mse_p, true_m = collect_one_sample(Exp_Main(ns_mse), MSE_SETTING, si)
    del ns_mse

    ns_lc = make_lc_args()
    _set_device(ns_lc, args.gpu, args.cpu)
    _, lc_p, true_lc = collect_one_sample(Exp_Main(ns_lc), LC_SETTING, si)
    del ns_lc

    ns_sar = make_sar_args()
    _set_device(ns_sar, args.gpu, args.cpu)
    _, sar_p, true_s = collect_one_sample(Exp_Main(ns_sar), SAR_SETTING, si)
    del ns_sar

    if not (np.allclose(true_m, true_lc) and np.allclose(true_m, true_s)):
        print("WARNING: ground truth mismatch across loaders", flush=True)

    n_ch = true_m.shape[-1]
    names = _channel_names(n_ch)
    gm = load_test_mse(MSE_SETTING)
    gl = load_test_mse(LC_SETTING)
    gs = load_test_mse(SAR_SETTING)
    title = (
        f"Weather pl=720 all channels | sample={si} | zoom={args.zoom_len}\n"
        f"test MSE: MSE={gm:.4f} | LC={gl:.4f} | SAR={gs:.4f}"
        if gm and gl and gs
        else f"Weather pl=720 all channels | sample={si}"
    )

    base = f"weather_pl720_sample{si}_allch"
    full_path = os.path.join(out_dir, f"{base}_zoom{args.zoom_len}.png")
    plot_all_channels(
        inp, true_m, mse_p, lc_p, sar_p, names, full_path, title, args.hist_len, args.zoom_len
    )

    npz_path = os.path.join(out_dir, f"{base}.npz")
    np.savez(
        npz_path,
        inputs=inp,
        true=true_m,
        mse=mse_p,
        lc=lc_p,
        sar=sar_p,
        sample_idx=si,
        channel_names=np.array(names, dtype=object),
    )
    print(f"saved -> {full_path}", flush=True)
    print(f"npz  -> {npz_path}", flush=True)


if __name__ == "__main__":
    main()
