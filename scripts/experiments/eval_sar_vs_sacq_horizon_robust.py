"""Horizon-Stratified Robustness: SAR vs SACQ (ETTh1, clean-trained checkpoints).

Compares only SACQ+logcosh (bidirectional future self-attn) vs SAR+logcosh (causal).
Reports MSE/MAE on near / mid / far forecast segments under:

  - tail_shift:  OT level shift on last 50%% of history (× train σ_OT)
  - tail_spike:  +kσ impulses confined to last ``tail_len`` history steps
  - scaffold:    Gaussian noise on far-half coarse future patches (decoder-targeted)

Outputs CSV + calls plot_sar_vs_sacq_horizon_robust (unless --no_plot).

Example::
    python scripts/experiments/eval_sar_vs_sacq_horizon_robust.py --mode all --split test
    python scripts/experiments/plot_sar_vs_sacq_horizon_robust.py --split test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from typing import Iterator

import numpy as np
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

SEQ_LEN = 512
OT_CH = 6
SHIFT_TAIL_FRAC = 0.5
TAIL_SPIKE_LEN = 128

SHIFT_PCTS = list(range(0, 22, 2))
TAIL_SPIKE_DENSITIES = [0.05, 0.10]
TAIL_SPIKE_SIGMAS = [3, 5, 7]
SCAFFOLD_EPSILONS = [0.05, 0.10, 0.15, 0.20]

MODEL_COLORS = {
    "sar_slc": "#2ca02c",
    "sacq_slc": "#9467bd",
}


def resolve_sar_sacq_models(pred_len: int) -> list[dict]:
    return [
        {
            "tag": "sar_slc",
            "label": "SAR+logcosh",
            "setting": (
                f"H1s{SEQ_LEN}p{pred_len}SARslc_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hSARslc_0"
            ),
            "kwargs": dict(
                model_id=f"H1s{SEQ_LEN}p{pred_len}SARslc",
                head_type="sacq_cross_attention",
                loss="logcosh_scaled",
                des="hSARslc",
                sacq_autoregressive=1,
            ),
        },
        {
            "tag": "sacq_slc",
            "label": "SACQ+logcosh",
            "setting": (
                f"H1s{SEQ_LEN}p{pred_len}SACQslc_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hSACQslc_0"
            ),
            "kwargs": dict(
                model_id=f"H1s{SEQ_LEN}p{pred_len}SACQslc",
                head_type="sacq_cross_attention",
                loss="logcosh_scaled",
                des="hSACQslc",
                sacq_autoregressive=0,
            ),
        },
    ]


def horizon_bounds(pred_len: int) -> dict[str, tuple[int, int]]:
    n_near = pred_len // 4
    n_mid = (pred_len * 3) // 4
    return {
        "near": (0, n_near),
        "mid": (n_near, n_mid),
        "far": (n_mid, pred_len),
    }


def segment_metrics(preds: np.ndarray, trues: np.ndarray, pred_len: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, (a, b) in horizon_bounds(pred_len).items():
        p = preds[:, a:b, :]
        t = trues[:, a:b, :]
        out[f"mse_{name}"] = float(np.mean((p - t) ** 2))
        out[f"mae_{name}"] = float(np.mean(np.abs(p - t)))
    out["mse_all"] = float(np.mean((preds - trues) ** 2))
    out["mae_all"] = float(np.mean(np.abs(preds - trues)))
    return out


def make_args(m: dict, pred_len: int):
    kw = dict(
        dsacq_enable=0,
        cross_attn_layers=2,
        train_epochs=50,
        batch_size=128,
        patience=20,
    )
    kw.update(m["kwargs"])
    return _etth1_base(pred_len, seq_len=SEQ_LEN, **kw)


def get_train_stds(exp: Exp_Main) -> np.ndarray:
    ds, _ = exp._get_data(flag="train")
    if hasattr(ds, "scaler") and hasattr(ds.scaler, "scale_"):
        return np.asarray(ds.scaler.scale_, dtype=np.float64)
    data = ds.data_x if hasattr(ds, "data_x") else None
    if data is None:
        return np.ones(7, dtype=np.float64)
    return np.std(data, axis=0, ddof=0).clip(min=1e-8)


def corruption_seed(base: int, corruption_name: str) -> int:
    digest = hashlib.md5(corruption_name.encode()).hexdigest()
    return base + int(digest[:8], 16) % 10000


def _shift_time_slice(tlen: int, tail_frac: float) -> slice:
    if tail_frac >= 1.0:
        return slice(None)
    t0 = max(0, int(round(tlen * (1.0 - tail_frac))))
    return slice(t0, None)


def corrupt_batch(
    batch_x: torch.Tensor,
    cfg: dict,
    ch_std: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    if not cfg or cfg.get("kind") == "clean":
        return batch_x
    x = batch_x.clone()
    bsz, tlen, nch = x.shape
    kind = cfg["kind"]

    if kind == "tail_shift":
        pct = cfg["pct"]
        tail_frac = float(cfg.get("tail_frac", SHIFT_TAIL_FRAC))
        t_sl = _shift_time_slice(tlen, tail_frac)
        ch = OT_CH
        for b in range(bsz):
            sign = float(rng.choice([-1.0, 1.0]))
            x[b, t_sl, ch] += sign * pct * ch_std[ch]

    elif kind == "tail_spike":
        mult = cfg["sigma_mult"]
        tail_len = int(cfg.get("tail_len", TAIL_SPIKE_LEN))
        density = float(cfg["density"])
        ch = OT_CH
        t_start = max(0, tlen - tail_len)
        span = tlen - t_start
        n_spike = max(1, int(round(span * density)))
        for b in range(bsz):
            idx = rng.choice(np.arange(t_start, tlen), size=n_spike, replace=False)
            for t_idx in idx:
                sign = float(rng.choice([-1.0, 1.0]))
                x[b, int(t_idx), ch] += sign * mult * ch_std[ch]

    return x


def _find_sacq_head(exp: Exp_Main):
    backbone = exp.model.model if hasattr(exp.model, "model") else exp.model
    head = getattr(backbone, "head", None)
    if head is None or not hasattr(head, "_compute_coarse_patches"):
        return None
    return head


@contextmanager
def scaffold_perturbation(head, cfg: dict | None, rng: np.random.Generator) -> Iterator[None]:
    if head is None or not cfg or cfg.get("kind") != "scaffold":
        yield
        return

    orig = head._compute_coarse_patches
    eps = float(cfg["epsilon"])
    far_frac = float(cfg.get("far_frac", 0.5))
    half = max(1, int(round(head.out_patch_num * far_frac)))

    def patched(x, kv):
        flat = orig(x, kv)
        out = flat.clone()
        g = torch.Generator(device=out.device)
        g.manual_seed(int(rng.integers(0, 2**31 - 1)))
        noise = torch.randn(
            out.shape[0],
            out.shape[1] - half,
            out.shape[2],
            generator=g,
            device=out.device,
            dtype=out.dtype,
        )
        out[:, half:, :] = out[:, half:, :] + noise * eps
        return out

    head._compute_coarse_patches = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        head._compute_coarse_patches = orig  # type: ignore[method-assign]


def tail_shift_corruptions() -> list[tuple[str, dict, float]]:
    items: list[tuple[str, dict, float]] = []
    for p in SHIFT_PCTS:
        key = "clean" if p == 0 else f"tailshift_{p}"
        if p == 0:
            items.append((key, {}, 0.0))
        else:
            items.append(
                (
                    key,
                    {"kind": "tail_shift", "pct": p / 100.0, "tail_frac": SHIFT_TAIL_FRAC},
                    float(p),
                )
            )
    return items


def tail_spike_corruptions() -> list[tuple[str, dict, float]]:
    items: list[tuple[str, dict, float]] = [("clean", {}, 0.0)]
    for dens in TAIL_SPIKE_DENSITIES:
        pct = int(round(dens * 100))
        for sig in TAIL_SPIKE_SIGMAS:
            key = f"tailspike_p{pct}_k{sig}"
            items.append(
                (
                    key,
                    {
                        "kind": "tail_spike",
                        "density": dens,
                        "sigma_mult": float(sig),
                        "tail_len": TAIL_SPIKE_LEN,
                    },
                    float(sig),
                )
            )
    return items


def scaffold_corruptions() -> list[tuple[str, dict, float]]:
    items: list[tuple[str, dict, float]] = [("clean", {}, 0.0)]
    for eps in SCAFFOLD_EPSILONS:
        key = f"scaffold_eps{eps:.2f}".replace(".", "p")
        items.append(
            (
                key,
                {"kind": "scaffold", "epsilon": eps, "far_frac": 0.5},
                eps,
            )
        )
    return items


CORRUPTION_MODES = {
    "tail_shift": tail_shift_corruptions,
    "tail_spike": tail_spike_corruptions,
    "scaffold": scaffold_corruptions,
}


@torch.no_grad()
def eval_split_horizon(
    exp: Exp_Main,
    history_cfg: dict,
    scaffold_cfg: dict | None,
    ch_std: np.ndarray,
    seed: int,
    split: str = "test",
) -> dict[str, float]:
    _, loader = exp._get_data(flag=split)
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len
    rng = np.random.default_rng(seed)
    head = _find_sacq_head(exp)

    preds, trues = [], []
    exp.model.eval()
    with scaffold_perturbation(head, scaffold_cfg, rng):
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x = corrupt_batch(batch_x, history_cfg, ch_std, rng)

            out, _ = exp._split_model_output(exp._forward(batch_x))
            pred_btc = _to_btc(exp, out, batch_y)
            if pred_btc is None:
                pred_btc = out

            preds.append(pred_btc[:, -pl:, f_dim:].detach().cpu().numpy())
            trues.append(batch_y[:, -pl:, f_dim:].detach().cpu().numpy())

    preds_arr = np.concatenate(preds, axis=0)
    trues_arr = np.concatenate(trues, axis=0)
    met = segment_metrics(preds_arr, trues_arr, pl)
    met["n_samples"] = int(preds_arr.shape[0])
    return met


def _row_base(m: dict, split: str, group: str, cname: str, severity: float) -> dict:
    return {
        "model_tag": m["tag"],
        "model_label": m["label"],
        "setting": m["setting"],
        "corruption_group": group,
        "corruption": cname,
        "severity": severity,
        "split": split,
    }


def run_group(args, out_dir: str, group: str) -> list[dict]:
    corruptions = CORRUPTION_MODES[group]()
    rows: list[dict] = []

    for m in resolve_sar_sacq_models(args.pred_len):
        print(f"\n=== {m['tag']} ({group}) ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)

        for cname, cfg, severity in corruptions:
            if group == "scaffold":
                hist_cfg = {}
                scaf_cfg = None if cname == "clean" else cfg
            else:
                hist_cfg = cfg
                scaf_cfg = None

            print(f"  {cname} ...", flush=True)
            met = eval_split_horizon(
                exp,
                history_cfg=hist_cfg,
                scaffold_cfg=scaf_cfg,
                ch_std=ch_std,
                seed=corruption_seed(args.seed, f"{group}:{cname}"),
                split=args.split,
            )
            row = {**_row_base(m, args.split, group, cname, severity), **met}
            rows.append(row)
            print(
                f"    all={met['mse_all']:.4f} near={met['mse_near']:.4f} "
                f"mid={met['mse_mid']:.4f} far={met['mse_far']:.4f}",
                flush=True,
            )
    return rows


def append_csv(csv_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(csv_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="SAR vs SACQ horizon-stratified robustness eval")
    parser.add_argument(
        "--out_dir",
        default="logs/etth1_sar_vs_sacq_horizon_robust",
        help="output directory under project root",
    )
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "tail_shift", "tail_spike", "scaffold"],
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to CSV instead of overwriting (per-group file still overwritten)",
    )
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    groups = list(CORRUPTION_MODES.keys()) if args.mode == "all" else [args.mode]
    all_rows: list[dict] = []

    meta = {
        "seq_len": SEQ_LEN,
        "pred_len": args.pred_len,
        "split": args.split,
        "seed": args.seed,
        "horizon_bounds": horizon_bounds(args.pred_len),
        "corruption_groups": groups,
        "tail_spike_len": TAIL_SPIKE_LEN,
        "shift_tail_frac": SHIFT_TAIL_FRAC,
        "models": [{k: m[k] for k in ("tag", "label", "setting")} for m in resolve_sar_sacq_models(args.pred_len)],
    }
    with open(os.path.join(out_dir, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for group in groups:
        rows = run_group(args, out_dir, group)
        group_csv = os.path.join(out_dir, f"results_{args.split}_{group}.csv")
        write_csv(group_csv, rows)
        print(f"saved -> {group_csv}", flush=True)
        all_rows.extend(rows)

    combined_csv = os.path.join(out_dir, f"results_{args.split}_horizon_stratified.csv")
    if args.append and os.path.isfile(combined_csv):
        append_csv(combined_csv, all_rows)
    else:
        write_csv(combined_csv, all_rows)
    print(f"saved -> {combined_csv}", flush=True)

    if not args.no_plot and all_rows:
        from scripts.experiments.plot_sar_vs_sacq_horizon_robust import plot_all

        plot_all(out_dir, args.split, args.pred_len)


if __name__ == "__main__":
    main()
