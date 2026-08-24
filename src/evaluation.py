"""Leakage-safe expanding-window evaluation and error diagnostics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import seasonal_mase_scale
from .models import MODEL_METADATA, forecast, model_names


def assert_no_temporal_leakage(train_timestamps: Iterable[int], target_timestamps: Iterable[int]) -> None:
    train = np.asarray(list(train_timestamps), dtype=int)
    target = np.asarray(list(target_timestamps), dtype=int)
    if train.size == 0 or target.size == 0:
        raise ValueError("Leakage check requires non-empty training and target timestamps.")
    if int(train.max()) >= int(target.min()):
        raise AssertionError(
            f"Temporal leakage: max(training_timestamp)={train.max()} is not less than "
            f"min(forecast_target_timestamp)={target.min()}."
        )


def period_label(period_index: int, seasonal_period: int = 4) -> str:
    year = (period_index - 1) // seasonal_period + 1
    quarter = (period_index - 1) % seasonal_period + 1
    return f"Y{year} Q{quarter}"


def expanding_window_evaluate(
    values: Iterable[float],
    models: Iterable[str] | None = None,
    initial_train_size: int = 8,
    max_horizon: int = 4,
    seasonal_period: int = 4,
) -> pd.DataFrame:
    """Evaluate every model on identical expanding origins and available future horizons."""
    series = np.asarray(list(values), dtype=float)
    models = list(models or model_names())
    if initial_train_size <= seasonal_period:
        raise ValueError("Initial training size must exceed one seasonal period for seasonal MASE.")
    if initial_train_size >= len(series):
        raise ValueError("Initial training size must leave at least one forecast target.")
    if max_horizon < 1:
        raise ValueError("Maximum horizon must be positive.")

    records: list[dict] = []
    for train_size in range(initial_train_size, len(series)):
        horizon = min(max_horizon, len(series) - train_size)
        train = series[:train_size]
        actual = series[train_size : train_size + horizon]
        train_timestamps = np.arange(1, train_size + 1)
        target_timestamps = np.arange(train_size + 1, train_size + horizon + 1)
        assert_no_temporal_leakage(train_timestamps, target_timestamps)
        scale = seasonal_mase_scale(train, seasonal_period)

        for model_name in models:
            predictions = forecast(model_name, train, horizon, seasonal_period)
            for step, (timestamp, observed, predicted) in enumerate(
                zip(target_timestamps, actual, predictions), start=1
            ):
                error = float(observed - predicted)
                denominator = abs(float(observed)) + abs(float(predicted))
                records.append(
                    {
                        "model": model_name,
                        "origin_period": int(train_size),
                        "origin_label": period_label(train_size, seasonal_period),
                        "train_start_period": 1,
                        "train_end_period": int(train_size),
                        "target_period": int(timestamp),
                        "target_label": period_label(int(timestamp), seasonal_period),
                        "target_quarter": int((timestamp - 1) % seasonal_period + 1),
                        "horizon": int(step),
                        "actual": float(observed),
                        "prediction": float(predicted),
                        "error": error,
                        "absolute_error": abs(error),
                        "squared_error": error**2,
                        "scaled_absolute_error": abs(error) / scale,
                        "smape_component": 200.0 * abs(error) / denominator if denominator else np.nan,
                        "mase_scale": scale,
                        "leakage_check": True,
                    }
                )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("Evaluation produced no forecast records.")
    return result


def model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = predictions.groupby("model", sort=False)
    rows: list[dict] = []
    for model_name, frame in grouped:
        metadata = MODEL_METADATA[model_name]
        rows.append(
            {
                "Model": model_name,
                "MAE": float(frame["absolute_error"].mean()),
                "RMSE": float(np.sqrt(frame["squared_error"].mean())),
                "MASE": float(frame["scaled_absolute_error"].mean()),
                "sMAPE": float(frame["smape_component"].mean()),
                "Mean_Error": float(frame["error"].mean()),
                "Number_of_Forecast_Origins": int(frame["origin_period"].nunique()),
                "Number_of_Predictions": int(len(frame)),
                "Complexity": metadata.complexity,
                "Strength": metadata.strength,
                "Limitation": metadata.limitation,
                "_complexity_order": metadata.complexity_order,
            }
        )
    comparison = pd.DataFrame(rows)
    for metric in ("MAE", "RMSE", "MASE"):
        comparison[f"{metric}_Rank"] = comparison[metric].rank(method="min", ascending=True)
    comparison["Average_Metric_Rank"] = comparison[["MAE_Rank", "RMSE_Rank", "MASE_Rank"]].mean(axis=1)
    comparison = comparison.sort_values(
        ["Average_Metric_Rank", "_complexity_order", "MASE", "MAE"], kind="stable"
    ).reset_index(drop=True)
    return comparison


def select_primary_model(comparison: pd.DataFrame) -> str:
    """Select by multi-metric rank, with lower complexity as an explicit tie-breaker."""
    if comparison.empty:
        raise ValueError("Model comparison is empty.")
    return str(comparison.iloc[0]["Model"])


def error_by_horizon(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = predictions[predictions["model"] == model_name]
    return (
        frame.groupby("horizon", as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            Mean_Error=("error", "mean"),
            Count=("error", "size"),
        )
        .sort_values("horizon")
    )


def error_by_quarter(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    frame = predictions[predictions["model"] == model_name]
    return (
        frame.groupby("target_quarter", as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            Mean_Error=("error", "mean"),
            Count=("error", "size"),
        )
        .sort_values("target_quarter")
    )


