"""ETTh1 SAR+logcosh: prediction + α + δ(corr) on one figure (per channel)."""
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
    _set_device,
    _to_btc,
    collect_all_preds,
    collect_one_sample,
    make_sar_args,
)


def _btc(exp, aux, aux_key, batch_y):
    t = aux.get(aux_key) if aux else None
    if t is None:
        return None
    out = _to_btc(exp, t, batch_y)
    if out is None:
        if t.shape[1] == batch_y.shape[2]:
            out = t.permute(0, 2, 1)
        else:
            out = t
    return out


def collect_one_sample_aux(
    exp: Exp_Main, setting: str, sample_idx: int, channel: int
) -> dict:
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"loaded {ckpt}", flush=True)

    bs = exp.args.batch_size
    target_batch, offset = sample_idx // bs, sample_idx % bs
    _, test_loader = exp._get_data(flag="test")
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if bi < target_batch:
                continue
            if bi > target_batch:
                raise IndexError(f"sample_idx={sample_idx} out of range")
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            out, aux = exp._split_model_output(exp._forward(batch_x))
            pred_btc = _to_btc(exp, out, batch_y)
            alpha_btc = _btc(exp, aux, "sacq_alpha_seq", batch_y)
            corr_btc = _btc(exp, aux, "sacq_corr_seq", batch_y)
            flat_btc = _btc(exp, aux, "sacq_flat_seq", batch_y)

            c = channel
            inp = batch_x[offset, :, c].detach().cpu().numpy()
            true = batch_y[offset, -pl:, f_dim + c].detach().cpu().numpy()
            pred = pred_btc[offset, -pl:, f_dim + c].detach().cpu().numpy()
            alpha = alpha_btc[offset, -pl:, f_dim + c].detach().cpu().numpy()
            delta = corr_btc[offset, -pl:, f_dim + c].detach().cpu().numpy()
            flat = flat_btc[offset, -pl:, f_dim + c].detach().cpu().numpy()
            return dict(
                inp=inp, true=true, pred=pred, alpha=alpha, delta=delta, flat=flat,
            )
    raise IndexError(f"sample_idx={sample_idx} not found")


def pick_sample(exp, setting, channel: int, mode: str) -> int:
    preds, trues = collect_all_preds(exp, setting)
    err = np.mean((preds[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    if mode == "max_err":
        return int(np.argmax(err))
    if mode == "median_err":
        return int(np.argsort(err)[len(err) // 2])
    return 0


def plot_pred_alpha_delta(
    data: dict,
    out_path: str,
    title: str,
    hist_len: int,
    zoom_len: int,
    patch_len: int = 16,
):
    inp = data["inp"]
    true = data["true"][:zoom_len]
    pred = data["pred"][:zoom_len]
    alpha = data["alpha"][:zoom_len]
    delta = data["delta"][:zoom_len]
    flat = data["flat"][:zoom_len]

    seq_len = len(inp)
    h0 = max(0, seq_len - hist_len)
    t_hist = np.arange(h0, seq_len)
    t_fut = np.arange(seq_len, seq_len + zoom_len)

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 8.5), sharex=False,
        gridspec_kw={"height_ratios": [2.4, 1.0, 1.0], "hspace": 0.28},
    )
    ax_pred, ax_a, ax_d = axes

    ax_pred.plot(t_hist, inp[h0:], color="#9e9e9e", lw=1.0, alpha=0.8, label="history")
    ax_pred.plot(t_fut, true, color="#1f77b4", lw=1.8, label="ground truth")
    ax_pred.plot(t_fut, flat, color="#9467bd", lw=1.2, ls=":", alpha=0.9, label="coarse (flat)")
    ax_pred.plot(t_fut, pred, color="#ff7f0e", lw=1.6, ls="--", label="SAR prediction")
    ax_pred.axvline(seq_len - 0.5, color="k", ls=":", lw=0.8, alpha=0.45)
    loc_mse = float(np.mean((pred - true) ** 2))
    ax_pred.set_ylabel("value (scaled)")
    ax_pred.set_title(title + f"\nlocal MSE={loc_mse:.4f}  (y ≈ flat + δ, δ = α·attn)")
    ax_pred.legend(loc="upper left", fontsize=9)
    ax_pred.grid(True, alpha=0.22)

    ax_a.plot(t_fut, alpha, color="#2ca02c", lw=1.4, drawstyle="steps-mid", label=r"$\alpha_t$")
    ax_a.set_ylabel(r"$\alpha$")
    ax_a.set_title(r"Gate $\alpha$ over forecast horizon", fontsize=10)
    ax_a.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(True, alpha=0.22)

    ax_d.plot(t_fut, delta, color="#d62728", lw=1.4, label=r"$\delta_t$ (corr = $\alpha$·attn)")
    ax_d.axhline(0, color="#666", lw=0.6, alpha=0.5)
    ax_d.set_ylabel(r"$\delta$")
    ax_d.set_xlabel("time step")
    ax_d.set_title(r"Correction $\delta$ over forecast horizon", fontsize=10)
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.grid(True, alpha=0.22)

    n_patch = max(1, zoom_len // patch_len)
    for ax in (ax_a, ax_d):
        for k in range(1, n_patch):
            x = seq_len + k * patch_len - 0.5
            if x < seq_len + zoom_len:
                ax.axvline(x, color="#ccc", ls=":", lw=0.5, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_test_mse(setting: str) -> float | None:
    p = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(json.load(f)["mse"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument(
        "--setting",
        type=str,
        default="",
        help="default: H1s512p{pl}SARslc from etth1_sar_sl512_logcosh_scaled",
    )
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument("--pick", type=str, default="max_err", choices=["max_err", "median_err", "first"])
    parser.add_argument("--channel", type=int, default=6, help="6=OT")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth1_sar_sl512_logcosh_scaled/curves",
    )
    parser.add_argument("--hist_len", type=int, default=128)
    parser.add_argument("--zoom_len", type=int, default=240, help="forecast steps shown (720 full is dense)")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    pl = args.pred_len
    setting = args.setting or SAR_SETTINGS[pl]
    zoom_len = args.zoom_len if args.zoom_len > 0 else pl
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ns = make_sar_args(pl)
    _set_device(ns, args.gpu, args.cpu)
    exp = Exp_Main(ns)

    ch = args.channel
    names = _channel_names(7)
    ch_name = names[ch] if ch < len(names) else f"ch{ch}"

    if args.sample_idx < 0:
        si = pick_sample(exp, setting, ch, args.pick)
        note = f"auto sample={si} ({args.pick})"
    else:
        si = args.sample_idx
        note = f"sample={si}"

    data = collect_one_sample_aux(exp, setting, si, ch)
    gm = load_test_mse(setting)
    title = f"ETTh1 SAR pl={pl} sl=512 | {ch_name} | {note} | zoom={zoom_len}"
    if gm is not None:
        title += f" | test MSE={gm:.4f}"

    base = f"etth1_sar_pl{pl}_sample{si}_{ch_name}_alpha_delta"
    out_path = os.path.join(out_dir, f"{base}_zoom{zoom_len}.png")
    plot_pred_alpha_delta(data, out_path, title, args.hist_len, zoom_len, patch_len=ns.patch_len)
    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        **data,
        sample_idx=si,
        channel=ch,
        channel_name=ch_name,
        setting=setting,
    )
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
