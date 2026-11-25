"""Leakage-safe datasets for long-horizon forecasting."""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .scaling import StandardScaler


DATASET_FILES = {
    "etth1": "ETTh1.csv",
    "weather": "weather.csv",
    "electricity": "electricity.csv",
}

ETTH1_TRAIN_END = 12 * 30 * 24
ETTH1_VAL_END = ETTH1_TRAIN_END + 4 * 30 * 24
ETTH1_TEST_END = ETTH1_VAL_END + 4 * 30 * 24


def split_borders(
    dataset_name: str,
    total_rows: int,
) -> Tuple[int, int, int]:
    """Return exclusive train, validation, and test endpoints."""

    if dataset_name == "etth1":
        if total_rows < ETTH1_TEST_END:
            raise ValueError(
                f"ETTh1 requires at least {ETTH1_TEST_END} rows, "
                f"found {total_rows}"
            )
        return (
            ETTH1_TRAIN_END,
            ETTH1_VAL_END,
            ETTH1_TEST_END,
        )

    train_end = int(total_rows * 0.7)
    val_end = int(total_rows * 0.8)
    return train_end, val_end, total_rows


class BenchmarkDataset(Dataset):
    """Multivariate forecasting dataset with train-only scaling.

    Inputs and targets are channel-first:

    - values: [C, context_length]
    - target: [C, prediction_length]

    Validation and test inputs may use context immediately preceding the
    target split, but targets never cross split boundaries.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        dataset_name: str,
        split: str,
        context_length: int = 512,
        prediction_length: int = 96,
    ) -> None:
        super().__init__()

        dataset_name = dataset_name.lower()
        if dataset_name not in DATASET_FILES:
            raise ValueError(
                f"unknown dataset {dataset_name!r}; "
                f"expected one of {tuple(DATASET_FILES)}"
            )
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        self.csv_path = Path(csv_path).resolve()
        self.dataset_name = dataset_name
        self.split = split
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)

        frame = pd.read_csv(self.csv_path)
        if "date" not in frame.columns:
            raise ValueError(
                f"{self.csv_path.name} must contain a date column"
            )

        self.columns = [
            column for column in frame.columns
            if column != "date"
        ]
        raw = frame[self.columns].to_numpy(dtype=np.float64)

        train_end, val_end, test_end = split_borders(
            dataset_name,
            len(raw),
        )
        self.train_end = train_end
        self.val_end = val_end
        self.test_end = test_end
        self.n_vars = len(self.columns)

        benchmark_raw = raw[:test_end]
        if not np.isfinite(benchmark_raw).all():
            raise ValueError(
                f"{dataset_name} benchmark region contains "
                "non-finite values"
            )

        self.scaler = StandardScaler.fit(
            benchmark_raw[:train_end]
        )
        scaled = self.scaler.transform(
            benchmark_raw
        ).astype(np.float32, copy=False)

        borders: Dict[str, Tuple[int, int]] = {
            "train": (0, train_end),
            "val": (
                train_end - self.context_length,
                val_end,
            ),
            "test": (
                val_end - self.context_length,
                test_end,
            ),
        }

        self.start_row, self.end_row = borders[split]
        if self.start_row < 0:
            raise ValueError(
                f"{dataset_name}/{split} has insufficient context"
            )

        self.data = scaled[self.start_row:self.end_row]
        self.window_count = (
            len(self.data)
            - self.context_length
            - self.prediction_length
            + 1
        )

        if self.window_count <= 0:
            raise ValueError(
                f"{dataset_name}/{split} is too short for "
                f"context={self.context_length}, "
                f"prediction={self.prediction_length}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int):
        if index < 0:
            index += self.window_count
        if not 0 <= index < self.window_count:
            raise IndexError(index)

        x_begin = index
        x_end = x_begin + self.context_length
        y_end = x_end + self.prediction_length

        values = torch.from_numpy(
            self.data[x_begin:x_end].T.copy()
        )
        target = torch.from_numpy(
            self.data[x_end:y_end].T.copy()
        )

        global_x_begin = self.start_row + x_begin
        global_x_end = self.start_row + x_end
        global_y_end = self.start_row + y_end

        return {
            "values": values,
            "target": target,
            "x_begin": global_x_begin,
            "x_end": global_x_end,
            "y_end": global_y_end,
        }


class ETTh1Dataset(BenchmarkDataset):
    """Backward-compatible ETTh1 dataset interface."""

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        context_length: int = 512,
        prediction_length: int = 96,
    ) -> None:
        super().__init__(
            csv_path=csv_path,
            dataset_name="etth1",
            split=split,
            context_length=context_length,
            prediction_length=prediction_length,
        )
