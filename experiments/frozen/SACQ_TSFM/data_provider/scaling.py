"""Train-split-only feature standardization."""

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    """Per-channel standardization fitted only on training rows."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "StandardScaler":
        if train_values.ndim != 2:
            raise ValueError("train_values must have shape [T,C]")

        mean = train_values.mean(axis=0, keepdims=True)
        scale = train_values.std(axis=0, keepdims=True)

        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("non-finite statistics in training data")

        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean
