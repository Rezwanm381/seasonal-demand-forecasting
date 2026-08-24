"""Forecast accuracy metrics with explicit seasonal MASE scaling."""

from __future__ import annotations

import numpy as np


def _aligned(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    if actual_array.size == 0 or not np.isfinite(actual_array).all() or not np.isfinite(predicted_array).all():
        raise ValueError("Metric inputs must be non-empty and finite.")
    return actual_array, predicted_array


def mae(actual, predicted) -> float:
    actual_array, predicted_array = _aligned(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual, predicted) -> float:
    actual_array, predicted_array = _aligned(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_array - predicted_array))))


def seasonal_mase_scale(train, seasonal_period: int = 4) -> float:
    train_array = np.asarray(train, dtype=float).reshape(-1)
    if len(train_array) <= seasonal_period:
        raise ValueError("Seasonal MASE requires more than one seasonal cycle in training.")
    scale = float(np.mean(np.abs(train_array[seasonal_period:] - train_array[:-seasonal_period])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Seasonal MASE scale must be finite and positive.")
    return scale


def mase(actual, predicted, train, seasonal_period: int = 4) -> float:
    return mae(actual, predicted) / seasonal_mase_scale(train, seasonal_period)


def smape(actual, predicted) -> float:
    actual_array, predicted_array = _aligned(actual, predicted)
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    if np.any(denominator == 0):
        raise ValueError("sMAPE is undefined where actual and predicted are both zero.")
    return float(np.mean(200.0 * np.abs(actual_array - predicted_array) / denominator))


def mean_error(actual, predicted) -> float:
    actual_array, predicted_array = _aligned(actual, predicted)
    return float(np.mean(actual_array - predicted_array))

