"""Compact, transparent forecasting models for a very short seasonal series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    complexity: str
    complexity_order: int
    strength: str
    limitation: str


MODEL_METADATA: dict[str, ModelMetadata] = {
    "NAIVE": ModelMetadata(
        "NAIVE", "Very low", 1, "Transparent local-level benchmark", "Ignores quarterly seasonality"
    ),
    "SEASONAL_NAIVE": ModelMetadata(
        "SEASONAL_NAIVE",
        "Very low",
        1,
        "Required benchmark that preserves the latest seasonal pattern",
        "Cannot adapt within-cycle level or trend",
    ),
    "SEASONAL_MEAN": ModelMetadata(
        "SEASONAL_MEAN",
        "Low",
        2,
        "Stable and interpretable quarter-specific averages",
        "Assumes a fixed seasonal level and ignores trend",
    ),
    "TREND_SEASONAL_ADDITIVE": ModelMetadata(
        "TREND_SEASONAL_ADDITIVE",
        "Moderate",
        3,
        "Separates a linear trend from additive quarter effects",
        "Five coefficients are fragile with only 8-12 training observations",
    ),
    "TREND_SEASONAL_MULTIPLICATIVE": ModelMetadata(
        "TREND_SEASONAL_MULTIPLICATIVE",
        "Moderate",
        3,
        "Rebuilds the historical trend-times-seasonal-factor method",
        "Multiplicative structure and linear extrapolation are weakly supported",
    ),
}


def _as_series(train: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(train, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Training data must contain finite observations.")
    return values


def naive(train: np.ndarray | list[float], horizon: int, seasonal_period: int = 4) -> np.ndarray:
    values = _as_series(train)
    return np.repeat(values[-1], horizon).astype(float)


def seasonal_naive(
    train: np.ndarray | list[float], horizon: int, seasonal_period: int = 4
) -> np.ndarray:
    values = _as_series(train)
    if len(values) < seasonal_period:
        raise ValueError("Seasonal naive requires at least one complete seasonal cycle.")
    last_cycle = values[-seasonal_period:]
    return np.asarray([last_cycle[step % seasonal_period] for step in range(horizon)], dtype=float)


def seasonal_mean(
    train: np.ndarray | list[float], horizon: int, seasonal_period: int = 4
) -> np.ndarray:
    values = _as_series(train)
    if len(values) < seasonal_period:
        raise ValueError("Seasonal mean requires at least one complete seasonal cycle.")
    means = np.asarray([values[quarter::seasonal_period].mean() for quarter in range(seasonal_period)])
    return np.asarray(
        [means[(len(values) + step) % seasonal_period] for step in range(horizon)], dtype=float
    )


def _additive_design(periods: np.ndarray, seasonal_period: int) -> np.ndarray:
    quarters = (periods - 1) % seasonal_period
    columns = [np.ones(len(periods)), periods.astype(float)]
    columns.extend((quarters == quarter).astype(float) for quarter in range(1, seasonal_period))
    return np.column_stack(columns)


def trend_seasonal_additive(
    train: np.ndarray | list[float], horizon: int, seasonal_period: int = 4
) -> np.ndarray:
    values = _as_series(train)
    minimum = seasonal_period * 2
    if len(values) < minimum:
        raise ValueError(f"Additive trend-seasonal regression requires at least {minimum} observations.")
    periods = np.arange(1, len(values) + 1, dtype=int)
    design = _additive_design(periods, seasonal_period)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    future_periods = np.arange(len(values) + 1, len(values) + horizon + 1, dtype=int)
    return (_additive_design(future_periods, seasonal_period) @ coefficients).astype(float)


def trend_seasonal_multiplicative(
    train: np.ndarray | list[float], horizon: int, seasonal_period: int = 4
) -> np.ndarray:
    values = _as_series(train)
    minimum = seasonal_period * 2
    if len(values) < minimum:
        raise ValueError(
            f"Multiplicative trend-seasonal method requires at least {minimum} observations."
        )
    periods = np.arange(1, len(values) + 1, dtype=float)
    design = np.column_stack([np.ones(len(periods)), periods])
    intercept, slope = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted_trend = intercept + slope * periods
    if np.any(fitted_trend <= 0):
        raise ValueError("Multiplicative seasonal ratios require a strictly positive fitted trend.")
    ratios = values / fitted_trend
    raw_factors = np.asarray([ratios[q::seasonal_period].mean() for q in range(seasonal_period)])
    factors = raw_factors / raw_factors.mean()
    future_periods = np.arange(len(values) + 1, len(values) + horizon + 1, dtype=float)
    future_trend = intercept + slope * future_periods
    forecasts = [
        future_trend[step] * factors[(len(values) + step) % seasonal_period]
        for step in range(horizon)
    ]
    return np.asarray(forecasts, dtype=float)


MODEL_FUNCTIONS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "NAIVE": naive,
    "SEASONAL_NAIVE": seasonal_naive,
    "SEASONAL_MEAN": seasonal_mean,
    "TREND_SEASONAL_ADDITIVE": trend_seasonal_additive,
    "TREND_SEASONAL_MULTIPLICATIVE": trend_seasonal_multiplicative,
}


def forecast(
    model_name: str,
    train: np.ndarray | list[float],
    horizon: int,
    seasonal_period: int = 4,
) -> np.ndarray:
    if model_name not in MODEL_FUNCTIONS:
        raise KeyError(f"Unknown model: {model_name}")
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")
    predictions = MODEL_FUNCTIONS[model_name](np.asarray(train, dtype=float), horizon, seasonal_period)
    if len(predictions) != horizon or not np.isfinite(predictions).all():
        raise ValueError(f"{model_name} returned invalid forecasts.")
    return predictions


def model_names() -> list[str]:
    return list(MODEL_FUNCTIONS)

