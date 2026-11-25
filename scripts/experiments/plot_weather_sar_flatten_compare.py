"""Plot ground truth vs full SAR vs Flatten-only (sacq_flat_seq) on Weather test set."""
from __future__ import annotations

import argparse
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

from eval_sacq_flatten_only import make_args_from_setting

DEFAULT_SETTING = "Ws512p720SARslc_PatchTST_weather_sl512_pl720_wSARslc_0"


def _to_btc(exp, tensor, batch_y):
    out = exp._to_btc(tensor, batch_y)
    if out is not None:
        return out
    if tensor.shape[1] == batch_y.shape[2]:
        return tensor.permute(0, 2, 1)
    return tensor


def collect_series(exp, max_batches=None):
    _, test_loader = exp._get_data(flag="test")
    sar_preds, flat_preds, trues, inputs = [], [], [], []
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, _, _) in enumerate(test_loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            sar_out, aux = exp._split_model_output(exp._forward(batch_x))
            sar_btc = _to_btc(exp, sar_out, batch_y)
            flat = aux.get("sacq_flat_seq") if aux else None
            if flat is None:
                raise RuntimeError("missing sacq_flat_seq in aux")
            flat_btc = _to_btc(exp, flat, batch_y)

            sar_preds.append(sar_btc[:, -pl:, f_dim:].detach().cpu().numpy())
            flat_preds.append(flat_btc[:, -pl:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -pl:, f_dim:].detach().cpu().numpy())
            inputs.append(batch_x.detach().cpu().numpy())

    sar_preds = np.concatenate(sar_preds, axis=0)
    flat_preds = np.concatenate(flat_preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    return inputs, sar_preds, flat_preds, trues


def pick_sample_index(sar_preds, flat_preds, trues, channel: int, mode: str) -> int:
    err_sar = np.mean((sar_preds[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    err_flat = np.mean((flat_preds[:, :, channel] - trues[:, :, channel]) ** 2, axis=1)
    gain = err_flat - err_sar
    if mode == "max_gain":
        return int(np.argmax(gain))
    if mode == "max_flat_err":
        return int(np.argmax(err_flat))
    return 0


def plot_compare(
    inp_hist,
    true_fut,
    sar_fut,
    flat_fut,
    out_path: str,
    title: str,
    zoom_len: int | None = None,
):
    seq_len = len(inp_hist)
    pred_len = len(true_fut)
    if zoom_len is not None and zoom_len < pred_len:
        true_fut = true_fut[:zoom_len]
        sar_fut = sar_fut[:zoom_len]
        flat_fut = flat_fut[:zoom_len]
        pred_len = zoom_len
        title = title + f" (first {zoom_len} steps)"

    t_in = np.arange(seq_len)
    t_fut = np.arange(seq_len, seq_len + pred_len)

    mse_sar = float(np.mean((sar_fut - true_fut) ** 2))
    mse_flat = float(np.mean((flat_fut - true_fut) ** 2))

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(t_in, inp_hist, color="#9e9e9e", linewidth=1.0, alpha=0.75, label="history")
    ax.plot(t_fut, true_fut, color="#1f77b4", linewidth=2.0, label="ground truth")
    ax.plot(
        t_fut,
        sar_fut,
        color="#d62728",
        linewidth=1.6,
        linestyle="-",
        label=f"SAR full (MSE={mse_sar:.4f})",
    )
    ax.plot(
        t_fut,
        flat_fut,
        color="#ff7f0e",
        linewidth=1.6,
        linestyle="--",
        label=f"Flatten only (MSE={mse_flat:.4f})",
    )
    ax.axvline(seq_len - 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("time step")
    ax.set_ylabel("value (scaled)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default=DEFAULT_SETTING)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/weather_sar_sl512_logcosh_scaled/curves",
    )
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument(
        "--pick",
        type=str,
        default="max_gain",
        choices=["max_gain", "max_flat_err", "first"],
        help="auto sample when sample_idx=-1",
    )
    parser.add_argument("--channel", type=int, default=20, help="OT column index for weather")
    parser.add_argument("--zoom_len", type=int, default=240, help="also save zoomed plot; 0=skip")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    ns = make_args_from_setting(args.setting, gpu=args.gpu)
    ns.pred_len = args.pred_len

    exp = Exp_Main(ns)
    ckpt = os.path.join(ns.checkpoints, args.setting, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    print(f"loaded {ckpt}")

    inputs, sar_preds, flat_preds, trues = collect_series(exp)
    ch = min(args.channel, sar_preds.shape[-1] - 1)

    if args.sample_idx < 0:
        si = pick_sample_index(sar_preds, flat_preds, trues, ch, args.pick)
        note = f"auto sample={si} ({args.pick}, ch={ch})"
    else:
        si = min(args.sample_idx, len(sar_preds) - 1)
        note = f"sample={si}, ch={ch}"

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    inp_ch = inputs[si, :, ch]
    true_f = trues[si, :, ch]
    sar_f = sar_preds[si, :, ch]
    flat_f = flat_preds[si, :, ch]

    base = f"Ws512p720_sample{si}_ch{ch}"
    title = (
        f"Weather pl={args.pred_len} | {args.setting}\n"
        f"({note}) | test MSE SAR=0.3153 Flatten=0.3426 (global)"
    )
    full_path = os.path.join(out_dir, f"{base}_full.png")
    plot_compare(inp_ch, true_f, sar_f, flat_f, full_path, title)

    if args.zoom_len and args.zoom_len > 0:
        zoom_path = os.path.join(out_dir, f"{base}_zoom{args.zoom_len}.png")
        plot_compare(
            inp_ch, true_f, sar_f, flat_f, zoom_path, title, zoom_len=args.zoom_len
        )
        print(f"saved zoom -> {zoom_path}")

    np.savez(
        os.path.join(out_dir, f"{base}.npz"),
        inputs=inputs[si],
        sar=sar_preds[si],
        flat=flat_preds[si],
        true=trues[si],
    )
    print(f"saved full -> {full_path}")


if __name__ == "__main__":
    main()
