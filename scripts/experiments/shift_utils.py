"""Synthetic distribution-shift dataset generation and presets."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET_META: Dict[str, Dict[str, Any]] = {
    "ETTh1": {"data": "ETTh1", "enc_in": 7, "csv": "ETTh1.csv", "freq": "h"},
    "ETTh2": {"data": "ETTh2", "enc_in": 7, "csv": "ETTh2.csv", "freq": "h"},
    "ETTm2": {"data": "ETTm2", "enc_in": 7, "csv": "ETTm2.csv", "freq": "t"},
    "weather": {"data": "weather", "enc_in": 21, "csv": "weather.csv", "freq": "t"},
}

SHIFT_PRESETS: Dict[str, Dict[str, Any]] = {
    "stage_combo": {
        "shift_type": "stage_combo",
        "shift_ratio": 0.8,
        "shock_steps": 300,
        "recovery_steps": 200,
        "level_mult_peak": 1.5,
        "level_mult_steady": 1.15,
        "old_slope": 0.1,
        "slope_peak": -0.2,
        "slope_steady": -0.05,
    },
    "mild_stage": {
        "shift_type": "stage_combo",
        "shift_ratio": 0.8,
        "shock_steps": 200,
        "recovery_steps": 150,
        "level_mult_peak": 1.25,
        "level_mult_steady": 1.08,
        "old_slope": 0.1,
        "slope_peak": -0.12,
        "slope_steady": -0.03,
    },
    "level_only": {
        "shift_type": "level_only",
        "shift_ratio": 0.8,
        "shock_steps": 300,
        "recovery_steps": 200,
        "level_mult_peak": 1.45,
        "level_mult_steady": 1.12,
        "old_slope": 0.0,
    },
    "trend_only": {
        "shift_type": "trend_only",
        "shift_ratio": 0.8,
        "shock_steps": 300,
        "recovery_steps": 200,
        "old_slope": 0.1,
        "slope_peak": -0.25,
        "slope_steady": -0.06,
    },
    "spike_revert": {
        "shift_type": "spike_revert",
        "shift_ratio": 0.8,
        "shock_steps": 120,
        "recovery_steps": 180,
        "level_mult_peak": 1.6,
        "old_slope": 0.1,
    },
    "gradual_ramp": {
        "shift_type": "gradual_ramp",
        "shift_ratio": 0.72,
        "level_mult_end": 1.4,
        "slope_end": -0.18,
        "old_slope": 0.1,
    },
    "gradual_ramp_strong": {
        "shift_type": "gradual_ramp",
        "shift_ratio": 0.72,
        "level_mult_end": 1.65,
        "slope_end": -0.28,
        "old_slope": 0.1,
    },
}


def _apply_shift_block(
    block: np.ndarray,
    mult: np.ndarray,
    slope: np.ndarray,
    old_slope: float,
) -> np.ndarray:
    slope_diff = slope - float(old_slope)
    trend_delta = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(slope_diff[:-1], dtype=np.float32)]
    )
    return block * mult[:, None] + trend_delta[:, None]


def make_shifted_dataset(
    dataset: str,
    preset: str = "stage_combo",
    dst_name: Optional[str] = None,
    root_path: str = "./dataset/",
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, str, Dict[str, Any]]:
    if dataset not in DATASET_META:
        raise ValueError(f"Unknown dataset {dataset!r}")
    meta = DATASET_META[dataset]
    params = dict(SHIFT_PRESETS.get(preset, SHIFT_PRESETS["stage_combo"]))
    if overrides:
        params.update(overrides)

    src_csv = os.path.join(root_path, meta["csv"])
    if dst_name is None:
        dst_name = f"{dataset}_shift_{preset}.csv"
    dst_csv = os.path.join(root_path, dst_name)

    df = pd.read_csv(src_csv)
    n = len(df)
    shift_ratio = float(params.get("shift_ratio", 0.8))
    shift_idx = int(n * shift_ratio)
    numeric_cols = [c for c in df.columns if c != "date"]
    after_len = n - shift_idx
    shift_type = params.get("shift_type", "stage_combo")
    old_slope = float(params.get("old_slope", 0.1))
    shock_steps = max(1, min(int(params.get("shock_steps", 300)), after_len))
    recovery_steps = int(params.get("recovery_steps", 200))

    if shift_type == "level_only":
        level_mult_peak = float(params.get("level_mult_peak", 1.4))
        level_mult_steady = float(params.get("level_mult_steady", 1.1))
        mult = np.full(after_len, level_mult_steady, dtype=np.float32)
        slope = np.full(after_len, old_slope, dtype=np.float32)
        mult[:shock_steps] = level_mult_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            mult[shock_steps:rec_end] = level_mult_peak + (level_mult_steady - level_mult_peak) * ratio
    elif shift_type == "trend_only":
        slope_peak = float(params.get("slope_peak", -0.2))
        slope_steady = float(params.get("slope_steady", -0.05))
        mult = np.ones(after_len, dtype=np.float32)
        slope = np.full(after_len, slope_steady, dtype=np.float32)
        slope[:shock_steps] = slope_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            slope[shock_steps:rec_end] = slope_peak + (slope_steady - slope_peak) * ratio
    elif shift_type == "gradual_ramp":
        level_end = float(params.get("level_mult_end", 1.4))
        slope_end = float(params.get("slope_end", -0.18))
        ratio = np.linspace(0.0, 1.0, after_len, dtype=np.float32)
        mult = 1.0 + (level_end - 1.0) * ratio
        slope = old_slope + (slope_end - old_slope) * ratio
    elif shift_type == "spike_revert":
        level_mult_peak = float(params.get("level_mult_peak", 1.5))
        mult = np.ones(after_len, dtype=np.float32)
        slope = np.full(after_len, old_slope, dtype=np.float32)
        mult[:shock_steps] = level_mult_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            mult[shock_steps:rec_end] = level_mult_peak + (1.0 - level_mult_peak) * ratio
    else:
        level_mult_peak = float(params.get("level_mult_peak", 1.5))
        level_mult_steady = float(params.get("level_mult_steady", 1.15))
        slope_peak = float(params.get("slope_peak", -0.2))
        slope_steady = float(params.get("slope_steady", -0.05))
        mult = np.full(after_len, level_mult_steady, dtype=np.float32)
        slope = np.full(after_len, slope_steady, dtype=np.float32)
        mult[:shock_steps] = level_mult_peak
        slope[:shock_steps] = slope_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            mult[shock_steps:rec_end] = level_mult_peak + (level_mult_steady - level_mult_peak) * ratio
            slope[shock_steps:rec_end] = slope_peak + (slope_steady - slope_peak) * ratio

    shifted = df.copy()
    block = shifted.loc[shift_idx:, numeric_cols].values.astype(np.float32)
    shifted.loc[shift_idx:, numeric_cols] = _apply_shift_block(block, mult, slope, old_slope)
    os.makedirs(os.path.dirname(os.path.abspath(dst_csv)), exist_ok=True)
    shifted.to_csv(dst_csv, index=False)

    info = {
        "dataset": dataset,
        "preset": preset,
        "shift_type": shift_type,
        "shift_idx": shift_idx,
        "n_total": n,
        "shock_steps": shock_steps,
        "recovery_steps": recovery_steps,
        "dst_csv": dst_csv,
        "params": params,
    }
    return shift_idx, n, dst_csv, info


def shift_csv_name(dataset: str, preset: str) -> str:
    return f"{dataset}_shift_{preset}.csv"


def drift_strength_profile(
    shift_idx: int,
    n_total: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per absolute time index: level mult, slope, scalar drift strength (higher = stronger shift)."""
    after_len = n_total - shift_idx
    shift_type = params.get("shift_type", "stage_combo")
    old_slope = float(params.get("old_slope", 0.1))
    shock_steps = max(1, min(int(params.get("shock_steps", 300)), after_len))
    recovery_steps = int(params.get("recovery_steps", 200))

    mult = np.ones(after_len, dtype=np.float32)
    slope = np.full(after_len, old_slope, dtype=np.float32)
    if shift_type == "gradual_ramp":
        level_end = float(params.get("level_mult_end", 1.4))
        slope_end = float(params.get("slope_end", -0.18))
        ratio = np.linspace(0.0, 1.0, after_len, dtype=np.float32)
        mult = 1.0 + (level_end - 1.0) * ratio
        slope = old_slope + (slope_end - old_slope) * ratio
    elif shift_type == "level_only":
        level_mult_peak = float(params.get("level_mult_peak", 1.4))
        level_mult_steady = float(params.get("level_mult_steady", 1.1))
        mult = np.full(after_len, level_mult_steady, dtype=np.float32)
        mult[:shock_steps] = level_mult_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            mult[shock_steps:rec_end] = level_mult_peak + (level_mult_steady - level_mult_peak) * ratio
    elif shift_type == "trend_only":
        slope_peak = float(params.get("slope_peak", -0.2))
        slope_steady = float(params.get("slope_steady", -0.05))
        slope = np.full(after_len, slope_steady, dtype=np.float32)
        slope[:shock_steps] = slope_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            slope[shock_steps:rec_end] = slope_peak + (slope_steady - slope_peak) * ratio
    else:
        level_mult_peak = float(params.get("level_mult_peak", 1.5))
        level_mult_steady = float(params.get("level_mult_steady", 1.15))
        slope_peak = float(params.get("slope_peak", -0.2))
        slope_steady = float(params.get("slope_steady", -0.05))
        mult = np.full(after_len, level_mult_steady, dtype=np.float32)
        slope = np.full(after_len, slope_steady, dtype=np.float32)
        mult[:shock_steps] = level_mult_peak
        slope[:shock_steps] = slope_peak
        rec_end = min(shock_steps + max(recovery_steps, 0), after_len)
        if rec_end > shock_steps:
            ratio = np.linspace(0.0, 1.0, rec_end - shock_steps, dtype=np.float32)
            mult[shock_steps:rec_end] = level_mult_peak + (level_mult_steady - level_mult_peak) * ratio
            slope[shock_steps:rec_end] = slope_peak + (slope_steady - slope_peak) * ratio

    t_post = np.arange(shift_idx, n_total, dtype=np.int64)
    strength = (mult - 1.0) + np.abs(slope - old_slope)
    return t_post, mult, strength
