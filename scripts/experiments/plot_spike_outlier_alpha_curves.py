#!/usr/bin/env python3
"""Alpha curves under input spike contamination — spike-outlier-trained SAR models.

Uses checkpoints from logs/etth1_spike_outlier_pl720 (or *_5x). On clean test data,
optionally injects ±noise_mult·σ spikes in the input window, then plots α over the
forecast horizon to check whether the gate suppresses correction (filtering).
"""
from __future__ import annotations

import argparse
import os
import re
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
from scripts.experiments.plot_etth1_flatten_vs_sar_all_channels import (
    _etth1_base,
    _set_device,
    _to_btc,
)
from scripts.experiments.plot_etth1_sar_alpha_delta_curves import plot_pred_alpha_delta

OT_CH = 6
DEFAULT_VARIANT = "SAR+logcosh"


def load_variant_summary(path: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & (df["variant_tag"] == variant)].copy()
    df["outlier_pct"] = df["outlier_pct"].astype(float)
    return df.sort_values("outlier_pct")


def _des_from_setting(setting: str) -> str:
    m = re.search(r"_pl\d+_(.+)_0$", setting)
    if m:
        return m.group(1)
    return "hSARlc"


def make_exp_from_row(row: pd.Series, pred_len: int) -> Exp_Main:
    ns = _etth1_base(
        pred_len,
        model_id=row["task_id"],
        des=_des_from_setting(row["setting"]),
        head_type=row["head_type"],
        sacq_autoregressive=1,
        loss=row["loss"],
        data_path="ETTh1.csv",
        scaler_data_path="ETTh1.csv",
        short_setting=0,
    )
    return Exp_Main(ns)


def inject_input_spikes(
    batch_x: torch.Tensor,
    ch_std: np.ndarray,
    ot_ch: int = OT_CH,
    n_spikes: int = 12,
    noise_mult: float = 3.0,
    seed: int = 0,
    tail_only: bool = False,
    tail_frac: float = 0.35,
) -> torch.Tensor:
    """Add point spikes on OT in the encoder input (scaled space)."""
    x = batch_x.clone()
    rng = np.random.default_rng(seed)
    bsz, tlen, _ = x.shape
    amp = noise_mult * float(ch_std[ot_ch])
    for b in range(bsz):
        if tail_only:
            t0 = int(round(tlen * (1.0 - tail_frac)))
            cand = np.arange(t0, tlen)
        else:
            cand = np.arange(tlen)
        if len(cand) == 0:
            continue
        n = min(n_spikes, len(cand))
        pos = rng.choice(cand, size=n, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n)
        x[b, pos, ot_ch] += torch.tensor(
            signs * amp, device=x.device, dtype=x.dtype
        )
    return x


def _alpha_btc(exp: Exp_Main, aux, batch_y):
    t = aux.get("sacq_alpha_seq") if aux else None
    if t is None:
        return None
    return _to_btc(exp, t, batch_y)


@torch.no_grad()
def collect_alpha_curves(
    exp: Exp_Main,
    setting: str,
    ch_std: np.ndarray,
    inject: bool,
    max_batches: int,
    noise_mult: float,
    n_spikes: int,
    tail_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_alpha_per_step [pl], std_alpha_per_step [pl]) over test loader."""
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    pl = exp.args.pred_len
    f_dim = -1 if exp.args.features == "MS" else 0
    c = OT_CH
    sums = np.zeros(pl, dtype=np.float64)
    sums_sq = np.zeros(pl, dtype=np.float64)
    count = 0

    _, loader = exp._get_data(flag="test")
    for bi, (batch_x, batch_y, _, _) in enumerate(loader):
        if bi >= max_batches:
            break
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        if inject:
            batch_x = inject_input_spikes(
                batch_x,
                ch_std,
                n_spikes=n_spikes,
                noise_mult=noise_mult,
                seed=2021 + bi,
                tail_only=tail_only,
            )
        out, aux = exp._split_model_output(exp._forward(batch_x))
        alpha_btc = _alpha_btc(exp, aux, batch_y)
        if alpha_btc is None:
            raise RuntimeError("sacq_alpha_seq missing — need SAR head checkpoint")
        alpha = alpha_btc[:, -pl:, f_dim + c].detach().cpu().numpy()
        sums += alpha.sum(axis=0)
        sums_sq += (alpha ** 2).sum(axis=0)
        count += alpha.shape[0]

    mean = sums / max(count, 1)
    var = sums_sq / max(count, 1) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


@torch.no_grad()
def collect_one_sample(
    exp: Exp_Main,
    setting: str,
    sample_idx: int,
    ch_std: np.ndarray,
    inject: bool,
    noise_mult: float,
    n_spikes: int,
    tail_only: bool,
) -> dict:
    ckpt = os.path.join(exp.args.checkpoints, setting, "checkpoint.pth")
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    exp.model.eval()

    bs = exp.args.batch_size
    target_batch, offset = sample_idx // bs, sample_idx % bs
    _, loader = exp._get_data(flag="test")
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len
    c = OT_CH

    for bi, (batch_x, batch_y, _, _) in enumerate(loader):
        if bi < target_batch:
            continue
        if bi > target_batch:
            raise IndexError(f"sample_idx={sample_idx} out of range")
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        x_clean = batch_x.clone()
        if inject:
            batch_x = inject_input_spikes(
                batch_x,
                ch_std,
                n_spikes=n_spikes,
                noise_mult=noise_mult,
                seed=sample_idx,
                tail_only=tail_only,
            )
        out, aux = exp._split_model_output(exp._forward(batch_x))
        pred_btc = _to_btc(exp, out, batch_y)
        alpha_btc = _alpha_btc(exp, aux, batch_y)
        corr_btc = _to_btc(exp, aux.get("sacq_corr_seq"), batch_y)
        flat_btc = _to_btc(exp, aux.get("sacq_flat_seq"), batch_y)

        inp_clean = x_clean[offset, :, c].cpu().numpy()
        inp_used = batch_x[offset, :, c].cpu().numpy()
        return dict(
            inp=inp_used,
            inp_clean=inp_clean,
            true=batch_y[offset, -pl:, f_dim + c].cpu().numpy(),
            pred=pred_btc[offset, -pl:, f_dim + c].cpu().numpy(),
            alpha=alpha_btc[offset, -pl:, f_dim + c].cpu().numpy(),
            delta=corr_btc[offset, -pl:, f_dim + c].cpu().numpy(),
            flat=flat_btc[offset, -pl:, f_dim + c].cpu().numpy(),
            injected=inject,
        )
    raise IndexError(f"sample_idx={sample_idx} not found")


def plot_alpha_horizon_panel(
    curves: dict[float, dict[str, np.ndarray]],
    out_path: str,
    title: str,
    noise_mult: float,
):
    """curves[pct] = {clean_mean, clean_std, spike_mean, spike_std}"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pl = len(next(iter(curves.values()))["clean_mean"])
    xs = np.arange(pl)
    colors = {0.0: "#444444", 1.0: "#8c564b", 3.0: "#e377c2", 5.0: "#2ca02c"}
    for pct, d in sorted(curves.items()):
        c = colors.get(pct, None)
        lbl = f"train spike {int(pct)}%" if pct > 0 else "train clean"
        ax.plot(xs, d["spike_mean"], lw=1.8, color=c, label=f"{lbl} | test +{noise_mult:g}σ spikes")
        ax.fill_between(
            xs,
            d["spike_mean"] - d["spike_std"],
            d["spike_mean"] + d["spike_std"],
            color=c,
            alpha=0.12,
        )
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean gate $\alpha$ (OT)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_alpha_suppression(
    curves: dict[float, dict[str, np.ndarray]],
    out_path: str,
    title: str,
):
    """Fractional drop: (α_clean - α_spiked) / α_clean on test-time spikes."""
    pcts = sorted(curves.keys())
    near_n = 180
    far_start = 540
    pl = len(curves[pcts[0]]["clean_mean"])

    def block_delta(d, sl):
        a0 = d["clean_mean"][sl].mean()
        a1 = d["spike_mean"][sl].mean()
        return (a0 - a1) / (abs(a0) + 1e-9) * 100.0

    x = np.array(pcts)
    near_sup = [block_delta(curves[p], slice(0, near_n)) for p in pcts]
    far_sup = [block_delta(curves[p], slice(far_start, pl)) for p in pcts]
    all_sup = [
        (
            curves[p]["clean_mean"].mean() - curves[p]["spike_mean"].mean()
        )
        / (abs(curves[p]["clean_mean"].mean()) + 1e-9)
        * 100.0
        for p in pcts
    ]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, near_sup, "o-", lw=2, color="#d62728", label=f"Near [0,{near_n})")
    ax.plot(x, far_sup, "s-", lw=2, color="#1f77b4", label=f"Far [{far_start},{pl})")
    ax.plot(x, all_sup, "^--", lw=1.6, color="#9467bd", alpha=0.85, label="All horizon")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Train spike-outlier ratio (%)")
    ax.set_ylabel(r"$(\alpha_{\mathrm{clean}}-\alpha_{\mathrm{spiked}})/\alpha_{\mathrm{clean}}$ (%)")
    ax.set_title(
        title
        + "\n(positive ⇒ α drops under test spikes; negative ⇒ α rises — not suppressing correction)"
    )
    ax.set_xticks(pcts)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="logs/etth1_spike_outlier_pl720/summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="logs/etth1_spike_outlier_pl720/figures",
    )
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_batches", type=int, default=80)
    parser.add_argument("--noise_mult", type=float, default=3.0)
    parser.add_argument("--n_spikes", type=int, default=12)
    parser.add_argument("--tail_only", action="store_true", help="inject only in last 35% of history")
    parser.add_argument("--sample_idx", type=int, default=42)
    parser.add_argument("--hist_len", type=int, default=160)
    parser.add_argument("--zoom_len", type=int, default=240)
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    summary_path = os.path.join(PROJECT_ROOT, args.summary)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = load_variant_summary(summary_path, args.variant)
    if df.empty:
        raise SystemExit(f"No rows for variant={args.variant} in {summary_path}")

    pl = args.pred_len
    curves: dict[float, dict[str, np.ndarray]] = {}

    for _, row in df.iterrows():
        pct = float(row["outlier_pct"])
        print(f"\n=== train outlier {pct}% | {row['setting']} ===", flush=True)
        exp = make_exp_from_row(row, pl)
        _set_device(exp.args, args.gpu, args.cpu)
        ds, _ = exp._get_data(flag="train")
        ch_std = np.asarray(ds.scaler.scale_, dtype=np.float64)

        clean_m, clean_s = collect_alpha_curves(
            exp,
            row["setting"],
            ch_std,
            inject=False,
            max_batches=args.max_batches,
            noise_mult=args.noise_mult,
            n_spikes=args.n_spikes,
            tail_only=args.tail_only,
        )
        spike_m, spike_s = collect_alpha_curves(
            exp,
            row["setting"],
            ch_std,
            inject=True,
            max_batches=args.max_batches,
            noise_mult=args.noise_mult,
            n_spikes=args.n_spikes,
            tail_only=args.tail_only,
        )
        curves[pct] = dict(
            clean_mean=clean_m,
            clean_std=clean_s,
            spike_mean=spike_m,
            spike_std=spike_s,
        )
        print(
            f"  α mean all: clean={clean_m.mean():.4e} spiked={spike_m.mean():.4e} "
            f"drop={(clean_m.mean()-spike_m.mean())/ (abs(clean_m.mean())+1e-9)*100:.2f}%",
            flush=True,
        )

    nm = args.noise_mult
    inj_note = "tail" if args.tail_only else "full hist"
    plot_alpha_horizon_panel(
        curves,
        os.path.join(out_dir, "spike_outlier_alpha_horizon_test_spikes.png"),
        title=(
            f"ETTh1 pl={pl} {args.variant} | mean α(OT) on clean test\n"
            f"test-time: {args.n_spikes}× ±{nm:g}σ spikes ({inj_note})"
        ),
        noise_mult=nm,
    )
    plot_alpha_suppression(
        curves,
        os.path.join(out_dir, "spike_outlier_alpha_suppression_vs_train_pct.png"),
        title=f"ETTh1 pl={pl} {args.variant} | gate response to test-time spikes",
    )

    # Mechanism figure: model trained on 5% spikes, clean vs spiked input on one sample
    row5 = df[df["outlier_pct"] == 5.0].iloc[0]
    exp5 = make_exp_from_row(row5, pl)
    _set_device(exp5.args, args.gpu, args.cpu)
    ds, _ = exp5._get_data(flag="train")
    ch_std = np.asarray(ds.scaler.scale_, dtype=np.float64)
    d_clean = collect_one_sample(
        exp5, row5["setting"], args.sample_idx, ch_std, inject=False,
        noise_mult=nm, n_spikes=args.n_spikes, tail_only=args.tail_only,
    )
    d_spike = collect_one_sample(
        exp5, row5["setting"], args.sample_idx, ch_std, inject=True,
        noise_mult=nm, n_spikes=args.n_spikes, tail_only=args.tail_only,
    )
    plot_pred_alpha_delta(
        d_clean,
        os.path.join(out_dir, f"spike_outlier_alpha_sample{args.sample_idx}_clean_input.png"),
        f"Train 5% spike | clean test input | sample {args.sample_idx}",
        args.hist_len,
        args.zoom_len,
        patch_len=exp5.args.patch_len,
    )
    plot_pred_alpha_delta(
        d_spike,
        os.path.join(out_dir, f"spike_outlier_alpha_sample{args.sample_idx}_spiked_input.png"),
        f"Train 5% spike | test input +{nm:g}σ spikes | sample {args.sample_idx}",
        args.hist_len,
        args.zoom_len,
        patch_len=exp5.args.patch_len,
    )

    # Overlay α curves on same axes
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(min(args.zoom_len, pl))
    ax.plot(t, d_clean["alpha"][: len(t)], lw=1.6, color="#2ca02c", label="α | clean input")
    ax.plot(t, d_spike["alpha"][: len(t)], lw=1.6, color="#d62728", label=f"α | +{nm:g}σ spikes")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"$\alpha$ (OT)")
    ax.set_title(f"Train 5% spike SAR | sample {args.sample_idx} | α under input contamination")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    p_overlay = os.path.join(out_dir, f"spike_outlier_alpha_overlay_sample{args.sample_idx}.png")
    fig.savefig(p_overlay, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {p_overlay}", flush=True)


if __name__ == "__main__":
    main()
