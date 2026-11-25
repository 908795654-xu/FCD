"""Inject spike outliers into ETT-style CSV train / test segments."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ETT hour: 12 / 4 / 4 months
ETT_HOUR_TRAIN_ROWS = 12 * 30 * 24
ETT_HOUR_VAL_ROWS = 4 * 30 * 24
ETT_HOUR_TEST_ROWS = 4 * 30 * 24
ETT_HOUR_TEST_START = ETT_HOUR_TRAIN_ROWS + ETT_HOUR_VAL_ROWS


def _col_std(block: np.ndarray) -> np.ndarray:
    std = block.std(axis=0, ddof=0)
    return np.where(std < 1e-8, 1.0, std)


def _inject_spikes_block(
    block: np.ndarray,
    outlier_ratio: float,
    noise_mult: float,
    col_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Return corrupted block and number of injected spikes."""
    n_scalars = block.size
    n_out = int(round(n_scalars * outlier_ratio))
    if n_out <= 0:
        return block.copy(), 0

    flat_idx = rng.choice(n_scalars, size=n_out, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_out)
    out = block.copy()
    n_cols = block.shape[1]
    t_idx = flat_idx // n_cols
    c_idx = flat_idx % n_cols
    out[t_idx, c_idx] += signs * noise_mult * col_std[c_idx]
    return out, n_out


def inject_spike_outliers(
    src_csv: str,
    dst_csv: str,
    outlier_ratio: float = 0.01,
    noise_mult: float = 3.0,
    train_rows: int = ETT_HOUR_TRAIN_ROWS,
    seed: int = 2021,
) -> Dict[str, Any]:
    """Randomly corrupt `outlier_ratio` of train scalars with ±noise_mult×per-channel std."""
    df = pd.read_csv(src_csv)
    numeric_cols = [c for c in df.columns if c != "date"]
    train_block = df.loc[: train_rows - 1, numeric_cols].values.astype(np.float64)
    n_scalars = train_block.size
    col_std = _col_std(train_block)

    rng = np.random.default_rng(seed)
    block, n_out = _inject_spikes_block(
        train_block, outlier_ratio, noise_mult, col_std, rng
    )

    corrupted = df.copy()
    if n_out > 0:
        corrupted.loc[: train_rows - 1, numeric_cols] = block
    corrupted.to_csv(dst_csv, index=False)

    meta = {
        "mode": "train_only",
        "outlier_ratio": outlier_ratio,
        "noise_mult": noise_mult,
        "train_rows": train_rows,
        "n_scalars": n_scalars,
        "n_outliers": n_out,
        "seed": seed,
        "col_std": {numeric_cols[i]: float(col_std[i]) for i in range(len(numeric_cols))},
    }
    meta_path = dst_csv.replace(".csv", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def inject_spike_outliers_bilateral(
    src_csv: str,
    dst_csv: str,
    train_outlier_ratio: float = 0.05,
    test_outlier_ratio: float = 0.05,
    noise_mult: float = 3.0,
    train_rows: int = ETT_HOUR_TRAIN_ROWS,
    test_start: int = ETT_HOUR_TEST_START,
    test_rows: int = ETT_HOUR_TEST_ROWS,
    train_seed: int = 2021,
    test_seed: int = 2022,
) -> Dict[str, Any]:
    """Corrupt train and test CSV segments independently (val left clean).

    Scaler statistics should still be fit on clean train data (``scaler_data_path``).
    Test MSE is computed against polluted future labels in the CSV.
    """
    df = pd.read_csv(src_csv)
    numeric_cols = [c for c in df.columns if c != "date"]
    test_end = test_start + test_rows

    train_block = df.loc[: train_rows - 1, numeric_cols].values.astype(np.float64)
    test_block = df.loc[test_start:test_end - 1, numeric_cols].values.astype(np.float64)

    train_std = _col_std(train_block)
    test_std = _col_std(test_block)

    train_rng = np.random.default_rng(train_seed)
    test_rng = np.random.default_rng(test_seed)
    train_corrupted, train_n_out = _inject_spikes_block(
        train_block, train_outlier_ratio, noise_mult, train_std, train_rng
    )
    test_corrupted, test_n_out = _inject_spikes_block(
        test_block, test_outlier_ratio, noise_mult, test_std, test_rng
    )

    corrupted = df.copy()
    if train_n_out > 0:
        corrupted.loc[: train_rows - 1, numeric_cols] = train_corrupted
    if test_n_out > 0:
        corrupted.loc[test_start:test_end - 1, numeric_cols] = test_corrupted
    corrupted.to_csv(dst_csv, index=False)

    meta = {
        "mode": "train_test_both",
        "train_outlier_ratio": train_outlier_ratio,
        "test_outlier_ratio": test_outlier_ratio,
        "noise_mult": noise_mult,
        "train_rows": train_rows,
        "test_start": test_start,
        "test_rows": test_rows,
        "train_n_scalars": int(train_block.size),
        "test_n_scalars": int(test_block.size),
        "train_n_outliers": train_n_out,
        "test_n_outliers": test_n_out,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "train_col_std": {
            numeric_cols[i]: float(train_std[i]) for i in range(len(numeric_cols))
        },
        "test_col_std": {
            numeric_cols[i]: float(test_std[i]) for i in range(len(numeric_cols))
        },
    }
    meta_path = dst_csv.replace(".csv", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def spike_csv_name(outlier_pct: float, noise_mult: float = 3.0) -> str:
    pct_tag = int(outlier_pct) if outlier_pct == int(outlier_pct) else outlier_pct
    mult_tag = int(noise_mult) if noise_mult == int(noise_mult) else noise_mult
    return f"ETTh1_spike_{mult_tag}x_p{pct_tag}.csv"


def bilateral_spike_csv_name(
    train_pct: float,
    test_pct: float,
    noise_mult: float = 3.0,
) -> str:
    mult_tag = int(noise_mult) if noise_mult == int(noise_mult) else noise_mult
    tr = int(train_pct) if train_pct == int(train_pct) else train_pct
    te = int(test_pct) if test_pct == int(test_pct) else test_pct
    if train_pct == test_pct:
        return f"ETTh1_spike_{mult_tag}x_p{tr}_both.csv"
    return f"ETTh1_spike_{mult_tag}x_tr{tr}_te{te}_both.csv"


def bilateral_spike_tag(train_pct: float, test_pct: float) -> str:
    tr = int(train_pct) if train_pct == int(train_pct) else train_pct
    te = int(test_pct) if test_pct == int(test_pct) else test_pct
    if train_pct == test_pct:
        return f"spp{tr}both"
    return f"sptr{tr}te{te}both"


def ensure_spike_dataset(
    dataset_dir: str,
    outlier_pct: float,
    noise_mult: float = 3.0,
    seed: int = 2021,
    src_name: str = "ETTh1.csv",
    force: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Return data_path basename; create corrupted CSV if needed."""
    ratio = outlier_pct / 100.0
    if ratio <= 0:
        return src_name, {"outlier_ratio": 0.0, "noise_mult": noise_mult}

    dst_name = spike_csv_name(outlier_pct, noise_mult)
    dst_csv = os.path.join(dataset_dir, dst_name)
    src_csv = os.path.join(dataset_dir, src_name)
    meta_path = dst_csv.replace(".csv", ".meta.json")
    need_build = force or not os.path.exists(dst_csv)
    if not need_build and os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if (
            abs(float(old.get("outlier_ratio", -1)) - ratio) > 1e-9
            or abs(float(old.get("noise_mult", -1)) - noise_mult) > 1e-9
            or int(old.get("seed", -1)) != seed
        ):
            need_build = True
    if need_build:
        meta = inject_spike_outliers(
            src_csv, dst_csv, outlier_ratio=ratio, noise_mult=noise_mult, seed=seed
        )
    else:
        with open(meta_path) as f:
            meta = json.load(f)
    return dst_name, meta


def ensure_bilateral_spike_dataset(
    dataset_dir: str,
    train_pct: float,
    test_pct: float,
    noise_mult: float = 3.0,
    train_seed: int = 2021,
    test_seed: int = 2022,
    src_name: str = "ETTh1.csv",
    force: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Return bilateral CSV basename; create if missing."""
    if train_pct <= 0 and test_pct <= 0:
        return src_name, {
            "mode": "clean",
            "train_outlier_ratio": 0.0,
            "test_outlier_ratio": 0.0,
            "noise_mult": noise_mult,
        }

    dst_name = bilateral_spike_csv_name(train_pct, test_pct, noise_mult)
    dst_csv = os.path.join(dataset_dir, dst_name)
    src_csv = os.path.join(dataset_dir, src_name)
    meta_path = dst_csv.replace(".csv", ".meta.json")
    train_ratio = train_pct / 100.0
    test_ratio = test_pct / 100.0

    need_build = force or not os.path.exists(dst_csv)
    if not need_build and os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if (
            abs(float(old.get("train_outlier_ratio", -1)) - train_ratio) > 1e-9
            or abs(float(old.get("test_outlier_ratio", -1)) - test_ratio) > 1e-9
            or abs(float(old.get("noise_mult", -1)) - noise_mult) > 1e-9
            or int(old.get("train_seed", -1)) != train_seed
            or int(old.get("test_seed", -1)) != test_seed
        ):
            need_build = True

    if need_build:
        meta = inject_spike_outliers_bilateral(
            src_csv,
            dst_csv,
            train_outlier_ratio=train_ratio,
            test_outlier_ratio=test_ratio,
            noise_mult=noise_mult,
            train_seed=train_seed,
            test_seed=test_seed,
        )
    else:
        with open(meta_path) as f:
            meta = json.load(f)
    return dst_name, meta
