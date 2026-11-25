"""Test whether shifting α (or correction) along forecast horizon improves MSE (lag diagnosis)."""
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main
from scripts.experiments.plot_sacq_curves import make_args
from utils.metrics import metric

SETTING = "TwS2Zc01A01g_PatchTST_ETTh2_sl512_pl720_hA01g_0"
PRED_LEN = 720
MAX_SHIFT = 24  # ± steps on forecast axis


def collect_decomposed(exp, max_batches=None):
    _, loader = exp._get_data(flag="test")
    flats, corrs, alphas, preds, trues = [], [], [], [], []
    f_dim = -1 if exp.args.features == "MS" else 0

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, _, _) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            out, aux = exp._split_model_output(exp._forward(batch_x))
            out = out[:, -PRED_LEN:, f_dim:]
            batch_y = batch_y[:, -PRED_LEN:, f_dim:]

            def btc(key):
                t = aux.get(key)
                if t is None:
                    return None
                x = exp._to_btc(t, batch_y)
                if x is None:
                    if t.shape[1] == batch_y.shape[2]:
                        x = t.permute(0, 2, 1)
                    else:
                        x = t
                return x.detach().cpu().numpy()

            flats.append(btc("sacq_flat_seq"))
            corrs.append(btc("sacq_corr_seq"))
            alphas.append(btc("sacq_alpha_seq"))
            preds.append(out.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    flat = np.concatenate(flats, axis=0)
    corr = np.concatenate(corrs, axis=0)
    alpha = np.concatenate(alphas, axis=0)
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    return flat, corr, alpha, pred, true


def roll_time(x, shift, pad_mode="edge"):
    """shift > 0: use values from the left (earlier forecast steps)."""
    if shift == 0:
        return x.copy()
    out = np.empty_like(x)
    T = x.shape[1]
    if shift > 0:
        out[:, :shift, :] = x[:, :1, :] if pad_mode == "edge" else 0
        out[:, shift:, :] = x[:, :-shift, :]
    else:
        s = -shift
        out[:, :-s, :] = x[:, s:, :]
        out[:, -s:, :] = x[:, -1:, :] if pad_mode == "edge" else 0
    return out


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def eval_shifts(flat_d, alpha, true, pred_model, shifts):
    """Use denorm-space flat and effective correction (pred - flat); RevIN denorms flat/corr separately."""
    eps = 1e-8
    corr_eff = pred_model - flat_d
    attn_equiv = corr_eff / np.maximum(alpha, eps)

    rows = []
    base = mse(pred_model, true)
    rows.append({"shift": 0, "mode": "model", "mse": base, "delta_pct": 0.0})
    rows.append({
        "shift": 0,
        "mode": "flat+corr_eff",
        "mse": mse(flat_d + corr_eff, true),
        "delta_pct": 0.0,
    })

    for s in shifts:
        if s == 0:
            continue
        a_sh = roll_time(alpha, s)
        p_alpha = flat_d + a_sh * attn_equiv
        rows.append({
            "shift": s,
            "mode": "alpha_only",
            "mse": mse(p_alpha, true),
            "delta_pct": (mse(p_alpha, true) - base) / base * 100,
        })

        c_sh = roll_time(corr_eff, s)
        p_corr = flat_d + c_sh
        rows.append({
            "shift": s,
            "mode": "corr_only",
            "mse": mse(p_corr, true),
            "delta_pct": (mse(p_corr, true) - base) / base * 100,
        })

    return rows, base, corr_eff, attn_equiv


def main():
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", SETTING, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        print("missing", ckpt)
        return

    exp = Exp_Main(make_args(PRED_LEN, SETTING, 0))
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))

    flat, corr, alpha, pred, true = collect_decomposed(exp)
    corr_eff = pred - flat
    print(f"model pred MSE={mse(pred, true):.6f}")
    print(f"flat+corr(aux) MSE={mse(flat+corr, true):.6f}  (mismatch OK: RevIN denorm per branch)")
    print(f"flat+corr_eff MSE={mse(flat+corr_eff, true):.6f}")
    print(f"alpha mean={alpha.mean():.4f} std={alpha.std():.4f}  |corr_eff| mean={np.abs(corr_eff).mean():.4f}")

    shifts = list(range(-MAX_SHIFT, MAX_SHIFT + 1))
    rows, base, corr_eff, attn_equiv = eval_shifts(flat, alpha, true, pred, shifts)

    out_dir = os.path.join(PROJECT_ROOT, "logs/etth2_alpha_lr_boost")
    os.makedirs(out_dir, exist_ok=True)
    report = os.path.join(out_dir, "alpha_shift_pl720.txt")

    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    with open(report, "w") as f:
        f.write(f"setting={SETTING}\n")
        f.write(f"baseline MSE (model)={base:.6f}\n")
        f.write("shift>0: α/corr 向更晚的预测步平移（用更早时刻的值）\n\n")
        for mode, rs in by_mode.items():
            rs = sorted(rs, key=lambda x: x["mse"])
            f.write(f"=== {mode} (best first) ===\n")
            for r in rs[:8]:
                f.write(
                    f"  shift={r['shift']:+4d}  MSE={r['mse']:.6f}  "
                    f"Δ={r['delta_pct']:+.3f}%\n"
                )
            f.write("\n")

    print(f"wrote {report}")
    for mode in ("alpha_only", "corr_only"):
        rs = sorted(by_mode[mode], key=lambda x: x["mse"])
        best = rs[0]
        print(
            f"best {mode}: shift={best['shift']:+d} "
            f"MSE={best['mse']:.6f} ({best['delta_pct']:+.3f}%)"
        )

    eps = 1e-8
    numer = (true - flat) * attn_equiv
    denom = attn_equiv ** 2 + eps
    alpha_star = np.clip(numer / denom, 0, 1)
    pred_oracle = flat + alpha_star * attn_equiv
    mse_oracle = mse(pred_oracle, true)
    print(f"oracle per-step alpha* (clamped): MSE={mse_oracle:.6f} ({(mse_oracle-base)/base*100:+.3f}% vs model)")


if __name__ == "__main__":
  main()
