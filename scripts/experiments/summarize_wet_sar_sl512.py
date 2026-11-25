"""Print MSE/MAE table from wet_sar_sl512 summary.csv."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="logs/wet_sar_sl512/summary.csv")
    parser.add_argument("--out", default="logs/wet_sar_sl512/RESULTS.md")
    args = parser.parse_args()

    path = os.path.join(PROJECT_ROOT, args.summary)
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    ok = ok.sort_values(["dataset", "pred_len"])

    lines = [
        "# Weather / ELC / Traffic — SAR sl512 (MSE loss)",
        "",
        "Config aligned with `logs/etth1_sar_sl512` (SACQ+AR, d_model=16, epochs=50, …).",
        "ELC/Traffic use smaller batch_size for GPU memory (16 / 8).",
        "",
        "## Test MSE / MAE",
        "",
        "| Dataset | pred_len | MSE | MAE |",
        "|---------|----------|-----|-----|",
    ]
    for _, r in ok.iterrows():
        lines.append(
            f"| {r['dataset']} | {int(r['pred_len'])} | {r['mse']:.6f} | {r['mae']:.6f} |"
        )

    fail = df[df["status"] != "ok"]
    if len(fail):
        lines.extend(["", "## Failed", ""])
        for _, r in fail.iterrows():
            lines.append(f"- {r['task_id']}: rc={r['return_code']}")

    text = "\n".join(lines) + "\n"
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
