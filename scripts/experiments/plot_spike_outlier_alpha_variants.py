#!/usr/bin/env python3
"""More α-vs-horizon figures in the style of spike_outlier_alpha_horizon_test_spikes.png."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.plot_spike_outlier_alpha_curves import (
    DEFAULT_VARIANT,
    collect_alpha_curves,
    load_variant_summary,
    make_exp_from_row,
)

TRAIN_PCTS = (0.0, 1.0, 3.0, 5.0)
COLORS = {0.0: "#444444", 1.0: "#8c564b", 3.0: "#e377c2", 5.0: "#2ca02c"}
CLEAN_TRAIN_SARLC = dict(
    task_id="H1s512p720SARslc",
    setting="H1s512p720SARslc_PatchTST_ETTh1_sl512_pl720_hSARslc_0",
    head_type="sacq_cross_attention",
    loss="logcosh_scaled",
    variant_tag="SAR+logcosh (clean train)",
    outlier_pct=-1.0,
)


def cache_key(pct, inject, noise_mult, n_spikes, tail_only, variant):
    return f"{variant}|pct={pct}|inj={int(inject)}|nm={noise_mult}|n={n_spikes}|tail={int(tail_only)}"


def load_cache(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    z = np.load(path, allow_pickle=True)
    return {str(k): dict(mean=z[f"{k}_mean"], std=z[f"{k}_std"]) for k in z["keys"]}


def save_cache(path: str, store: dict) -> None:
    keys = sorted(store.keys())
    payload = {"keys": np.array(keys, dtype=object)}
    for k in keys:
        payload[f"{k}_mean"] = store[k]["mean"]
        payload[f"{k}_std"] = store[k]["std"]
    np.savez(path, **payload)


def bin_curve(mean: np.ndarray, std: np.ndarray, bin_size: int):
    pl = len(mean)
    n_bins = pl // bin_size
    xs, m, s = [], [], []
    for i in range(n_bins):
        sl = slice(i * bin_size, (i + 1) * bin_size)
        xs.append((i + 0.5) * bin_size)
        m.append(mean[sl].mean())
        s.append(std[sl].mean())
    return np.array(xs), np.array(m), np.array(s)


def collect_one(
    row,
    pl,
    gpu,
    cpu,
    inject,
    noise_mult,
    n_spikes,
    tail_only,
    max_batches,
    variant,
    store,
):
    k = cache_key(float(row["outlier_pct"]), inject, noise_mult, n_spikes, tail_only, variant)
    if k in store:
        return k, store[k]
    exp = make_exp_from_row(row, pl)
    from scripts.experiments.plot_spike_outlier_alpha_curves import _set_device

    _set_device(exp.args, gpu, cpu)
    ds, _ = exp._get_data(flag="train")
    ch_std = np.asarray(ds.scaler.scale_, dtype=np.float64)
    mean, std = collect_alpha_curves(
        exp,
        row["setting"],
        ch_std,
        inject=inject,
        max_batches=max_batches,
        noise_mult=noise_mult,
        n_spikes=n_spikes,
        tail_only=tail_only,
    )
    store[k] = dict(mean=mean, std=std)
    print(f"  collected {k} mean={mean.mean():.4e}", flush=True)
    return k, store[k]


def fig_spiked_only(store, df, variant, nm, n_spikes, tail_only, out_path, title_extra=""):
    """Same as original horizon_test_spikes."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pl = 720
    for pct in TRAIN_PCTS:
        row = df[df["outlier_pct"] == pct].iloc[0]
        k = cache_key(pct, True, nm, n_spikes, tail_only, variant)
        d = store[k]
        c = COLORS[pct]
        lbl = f"train spike {int(pct)}%" if pct > 0 else "train clean"
        ax.plot(np.arange(pl), d["mean"], lw=1.8, color=c, label=f"{lbl} | test +{nm:g}σ")
        ax.fill_between(
            np.arange(pl), d["mean"] - d["std"], d["mean"] + d["std"], color=c, alpha=0.12
        )
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean gate $\alpha$ (OT)")
    inj = "tail hist" if tail_only else "full hist"
    ax.set_title(
        f"ETTh1 pl=720 {variant} | test +{nm:g}σ ({n_spikes} spikes, {inj}){title_extra}"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_clean_vs_spiked(store, df, variant, nm, n_spikes, tail_only, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pl = 720
    for pct in TRAIN_PCTS:
        row = df[df["outlier_pct"] == pct].iloc[0]
        c = COLORS[pct]
        lbl = f"train {int(pct)}%" if pct > 0 else "train clean"
        kc = cache_key(pct, False, nm, n_spikes, tail_only, variant)
        ks = cache_key(pct, True, nm, n_spikes, tail_only, variant)
        ax.plot(
            np.arange(pl), store[kc]["mean"], lw=1.2, color=c, ls="--", alpha=0.75,
            label=f"{lbl} | clean input",
        )
        ax.plot(
            np.arange(pl), store[ks]["mean"], lw=1.8, color=c, ls="-",
            label=f"{lbl} | +{nm:g}σ input",
        )
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean $\alpha$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_binned72(store, df, variant, nm, n_spikes, tail_only, out_path, inject=True):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pct in TRAIN_PCTS:
        k = cache_key(pct, inject, nm, n_spikes, tail_only, variant)
        xs, m, s = bin_curve(store[k]["mean"], store[k]["std"], 72)
        c = COLORS[pct]
        lbl = f"train {int(pct)}%" if pct > 0 else "train clean"
        ax.plot(xs, m, "o-", lw=2, ms=5, color=c, label=lbl)
        ax.fill_between(xs, m - s, m + s, color=c, alpha=0.15)
    ax.set_xlabel("Forecast step (bin center, size=72)")
    ax.set_ylabel(r"Mean $\alpha$ (OT)")
    mode = "spiked" if inject else "clean"
    ax.set_title(f"ETTh1 pl=720 {variant} | binned α | {mode} input +{nm:g}σ")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_ratio(store, df, variant, nm, n_spikes, tail_only, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    pl = 720
    for pct in TRAIN_PCTS:
        kc = cache_key(pct, False, nm, n_spikes, tail_only, variant)
        ks = cache_key(pct, True, nm, n_spikes, tail_only, variant)
        ratio = store[ks]["mean"] / np.maximum(store[kc]["mean"], 1e-12)
        ax.plot(np.arange(pl), ratio, lw=1.8, color=COLORS[pct], label=f"train {int(pct)}%")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"$\alpha_{\mathrm{spiked}}/\alpha_{\mathrm{clean}}$")
    ax.set_title(f"ETTh1 pl=720 {variant} | α boost from test +{nm:g}σ spikes")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_2x2_panels(store, df, variant, nm, n_spikes, tail_only, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    pl = 720
    for ax, pct in zip(axes.flat, TRAIN_PCTS):
        ks = cache_key(pct, True, nm, n_spikes, tail_only, variant)
        kc = cache_key(pct, False, nm, n_spikes, tail_only, variant)
        ax.plot(np.arange(pl), store[kc]["mean"], color="#9e9e9e", lw=1.2, ls="--", label="clean in")
        ax.plot(np.arange(pl), store[ks]["mean"], color=COLORS[pct], lw=1.8, label=f"+{nm:g}σ in")
        ax.set_title(f"Train spike {int(pct)}%" if pct > 0 else "Train clean", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("Forecast step")
    axes[0, 0].set_ylabel(r"Mean $\alpha$ (OT)")
    fig.suptitle(f"ETTh1 pl=720 {variant} | α by training pollution", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_sigma_sweep(store, row5, variant, n_spikes, tail_only, sigmas, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pl = 720
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sigmas)))
    for nm, col in zip(sigmas, cmap):
        k = cache_key(5.0, True, nm, n_spikes, tail_only, variant)
        ax.plot(np.arange(pl), store[k]["mean"], lw=1.8, color=col, label=f"+{nm:g}σ")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean $\alpha$ (OT)")
    ax.set_title(f"ETTh1 pl=720 | train 5% spike {variant} | test spike severity sweep")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="Test input spikes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_variants_train5(store, rows_by_variant, nm, n_spikes, tail_only, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pl = 720
    style = {
        "SAR+logcosh": ("#2ca02c", "-"),
        "SAR+MSE": ("#1f77b4", "-."),
        "SAR+S2": ("#9467bd", ":"),
    }
    for vtag, row in rows_by_variant.items():
        k = cache_key(5.0, True, nm, n_spikes, tail_only, vtag)
        if k not in store:
            continue
        col, ls = style.get(vtag, ("#333", "-"))
        ax.plot(np.arange(pl), store[k]["mean"], lw=2, color=col, ls=ls, label=vtag)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean $\alpha$ (OT)")
    ax.set_title(f"ETTh1 pl=720 | train 5% spike | test +{nm:g}σ | SAR variants")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def fig_clean_train_compare(store, df, nm, n_spikes, tail_only, out_path):
    """Spike-trained 0/5% vs clean-trained SAR+logcosh."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pl = 720
    series = [
        (0.0, "spike-train 0%", "#444"),
        (5.0, "spike-train 5%", "#2ca02c"),
    ]
    for pct, lbl, c in series:
        k = cache_key(pct, True, nm, n_spikes, tail_only, DEFAULT_VARIANT)
        ax.plot(np.arange(pl), store[k]["mean"], lw=1.8, color=c, label=f"{lbl}")
    k0 = cache_key(-1.0, True, nm, n_spikes, tail_only, CLEAN_TRAIN_SARLC["variant_tag"])
    if k0 in store:
        ax.plot(
            np.arange(pl), store[k0]["mean"], lw=2, color="#d62728", ls="--",
            label="clean-train SAR+logcosh",
        )
    ax.set_xlabel("Forecast step")
    ax.set_ylabel(r"Mean $\alpha$ (OT)")
    ax.set_title(f"ETTh1 pl=720 | test +{nm:g}σ | training regime comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="logs/etth1_spike_outlier_pl720/summary.csv")
    parser.add_argument("--out_dir", default="logs/etth1_spike_outlier_pl720/figures")
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max_batches", type=int, default=60)
    parser.add_argument("--noise_mult", type=float, default=3.0)
    parser.add_argument("--n_spikes", type=int, default=12)
    parser.add_argument("--tail_only", action="store_true")
    parser.add_argument("--replot_only", action="store_true")
    parser.add_argument("--include_clean_train", action="store_true")
    parser.add_argument("--include_sar_variants", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "alpha_variants_cache.npz")

    df = load_variant_summary(os.path.join(PROJECT_ROOT, args.summary), args.variant)
    store = load_cache(cache_path) if args.replot_only else {}

    nm, ns, tail = args.noise_mult, args.n_spikes, args.tail_only
    pl = args.pred_len

    if not args.replot_only:
        for _, row in df.iterrows():
            for inject in (False, True):
                collect_one(
                    row, pl, args.gpu, args.cpu, inject, nm, ns, False,
                    args.max_batches, args.variant, store,
                )
                collect_one(
                    row, pl, args.gpu, args.cpu, inject, nm, ns, True,
                    args.max_batches, args.variant, store,
                )
        row5 = df[df["outlier_pct"] == 5.0].iloc[0]
        for sig in (1.0, 2.0, 3.0, 5.0, 7.0):
            for tail_f in (False, True):
                collect_one(
                    row5, pl, args.gpu, args.cpu, True, sig, ns, tail_f,
                    args.max_batches, args.variant, store,
                )
        if args.include_sar_variants:
            full = pd.read_csv(os.path.join(PROJECT_ROOT, args.summary))
            full = full[full["status"] == "ok"]
            for vtag in ("SAR+MSE", "SAR+S2"):
                sub = full[(full["variant_tag"] == vtag) & (full["outlier_pct"] == 5.0)]
                if len(sub):
                    for tail_f in (False, True):
                        collect_one(
                            sub.iloc[0], pl, args.gpu, args.cpu, True, nm, ns, tail_f,
                            args.max_batches, vtag, store,
                        )
        if args.include_clean_train:
            row_ct = pd.Series(CLEAN_TRAIN_SARLC)
            for tail_f in (False, True):
                collect_one(
                    row_ct, pl, args.gpu, args.cpu, True, nm, ns, tail_f,
                    args.max_batches, CLEAN_TRAIN_SARLC["variant_tag"], store,
                )
        save_cache(cache_path, store)

    # --- figures ---
    fig_spiked_only(
        store, df, args.variant, nm, ns, False,
        os.path.join(out_dir, "alpha_horizon_v2_spiked_fullhist.png"),
    )
    fig_spiked_only(
        store, df, args.variant, nm, ns, True,
        os.path.join(out_dir, "alpha_horizon_v2_spiked_tail_only.png"),
        title_extra=" | tail-only spikes",
    )
    fig_clean_vs_spiked(
        store, df, args.variant, nm, ns, tail,
        os.path.join(out_dir, "alpha_horizon_v2_clean_vs_spiked.png"),
    )
    fig_binned72(
        store, df, args.variant, nm, ns, False,
        os.path.join(out_dir, "alpha_horizon_v2_binned72_spiked.png"),
        inject=True,
    )
    fig_binned72(
        store, df, args.variant, nm, ns, False,
        os.path.join(out_dir, "alpha_horizon_v2_binned72_clean_input.png"),
        inject=False,
    )
    fig_ratio(
        store, df, args.variant, nm, ns, tail,
        os.path.join(out_dir, "alpha_horizon_v2_ratio.png"),
    )
    fig_2x2_panels(
        store, df, args.variant, nm, ns, tail,
        os.path.join(out_dir, "alpha_horizon_v2_2x2_panels.png"),
    )
    row5 = df[df["outlier_pct"] == 5.0].iloc[0]
    for tail_lbl, tail_f in (("fullhist", False), ("tailonly", True)):
        fig_sigma_sweep(
            store, row5, args.variant, ns, tail_f, (1.0, 2.0, 3.0, 5.0, 7.0),
            os.path.join(out_dir, f"alpha_horizon_v2_sigma_sweep_train5_{tail_lbl}.png"),
        )
    if args.include_sar_variants or args.replot_only:
        full = pd.read_csv(os.path.join(PROJECT_ROOT, args.summary))
        full = full[(full["status"] == "ok") & (full["outlier_pct"] == 5.0)]
        rows = {
            r["variant_tag"]: r
            for _, r in full.iterrows()
            if r["variant_tag"] in ("SAR+logcosh", "SAR+MSE", "SAR+S2")
        }
        if rows:
            fig_variants_train5(
                store, rows, nm, ns, tail,
                os.path.join(out_dir, "alpha_horizon_v2_sar_variants_train5.png"),
            )
    if args.include_clean_train or args.replot_only:
        fig_clean_train_compare(
            store, df, nm, ns, tail,
            os.path.join(out_dir, "alpha_horizon_v2_clean_vs_spiketrain.png"),
        )


if __name__ == "__main__":
    main()
