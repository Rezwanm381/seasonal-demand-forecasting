"""Run the leakage-safe seasonal forecasting analysis from an explicit local input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import EXPECTED_OBSERVATIONS, FORECAST_HORIZON, INITIAL_TRAIN_SIZE, SEASONAL_PERIOD
from src.data_prep import (
    extract_historical_forecasts,
    file_sha256,
    prepare_csv_data,
    prepare_development_data,
    write_data_quality_report,
)
from src.evaluation import (
    error_by_horizon,
    error_by_quarter,
    expanding_window_evaluate,
    model_comparison,
    period_label,
    select_primary_model,
)
from src.models import forecast, model_names
from src.visualization import (
    actual_vs_backtest,
    error_horizon,
    final_point_forecast,
    historical_time_series,
    model_performance,
    rolling_origin_design,
    seasonal_pattern,
)


DEVELOPMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEVELOPMENT_DIR.parent
DEFAULT_OUTPUT = DEVELOPMENT_DIR / ".private_outputs"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def _reconcile_historical_methods(values: np.ndarray, source_workbook: Path) -> pd.DataFrame:
    """Reconcile two historical formulas without treating them as validation evidence."""
    workbook = extract_historical_forecasts(source_workbook)
    workbook["python_seasonal_mean"] = forecast(
        "SEASONAL_MEAN", values, FORECAST_HORIZON, SEASONAL_PERIOD
    )
    workbook["python_trend_multiplicative"] = forecast(
        "TREND_SEASONAL_MULTIPLICATIVE", values, FORECAST_HORIZON, SEASONAL_PERIOD
    )
    workbook["seasonal_mean_absolute_difference"] = np.abs(
        workbook["workbook_seasonal_mean"] - workbook["python_seasonal_mean"]
    )
    workbook["trend_multiplicative_absolute_difference"] = np.abs(
        workbook["workbook_trend_multiplicative"] - workbook["python_trend_multiplicative"]
    )
    if not np.allclose(
        workbook["workbook_seasonal_mean"], workbook["python_seasonal_mean"], atol=1e-8
    ):
        raise AssertionError("Python seasonal-mean forecast does not reconcile to the workbook.")
    if not np.allclose(
        workbook["workbook_trend_multiplicative"],
        workbook["python_trend_multiplicative"],
        atol=1e-8,
    ):
        raise AssertionError("Python trend-multiplicative forecast does not reconcile to the workbook.")
    return workbook


def _eda_summary(data: pd.DataFrame) -> dict:
    values = data["demand"].to_numpy(float)
    periods = data["period_index"].to_numpy(float)
    slope = float(
        np.linalg.lstsq(
            np.column_stack([np.ones(len(periods)), periods]), values, rcond=None
        )[0][1]
    )
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "linear_slope_per_period_full_sample_description_only": slope,
        "quarter_means": {
            f"Q{int(quarter)}": float(value)
            for quarter, value in data.groupby("quarter")["demand"].mean().items()
        },
        "year_means": {
            f"Y{int(year)}": float(value)
            for year, value in data.groupby("year")["demand"].mean().items()
        },
        "iqr_outlier_periods": data.loc[outlier_mask, "period_label"].tolist(),
        "decomposition_used": False,
        "decomposition_reason": "Only three seasonal cycles; decomposition would invite overinterpretation.",
    }


def _load_input(
    data_path: Path | None,
    source_workbook: Path | None,
    processed_path: Path,
) -> tuple[pd.DataFrame, object, Path, str, str, str]:
    if (data_path is None) == (source_workbook is None):
        raise ValueError("Provide exactly one of data_path or source_workbook.")
    if data_path is not None:
        data, audit = prepare_csv_data(
            data_path,
            processed_path,
            seasonal_period=SEASONAL_PERIOD,
            expected_observations=EXPECTED_OBSERVATIONS,
        )
        return data, audit, data_path, "CSV", "USER_SUPPLIED", "USER_RESPONSIBILITY"

    assert source_workbook is not None
    data, audit = prepare_development_data(
        source_workbook,
        processed_path,
        seasonal_period=SEASONAL_PERIOD,
        expected_observations=EXPECTED_OBSERVATIONS,
    )
    return (
        data,
        audit,
        source_workbook,
        "AUTHORIZED_INTERNAL_WORKBOOK",
        "COURSE_PROVIDED",
        "VERIFY_BEFORE_PUBLICATION",
    )


def run(
    *,
    data_path: Path | None,
    source_workbook: Path | None,
    output_dir: Path,
    processed_path: Path | None = None,
) -> dict:
    """Execute preprocessing, backtesting, model selection, and point forecasting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    internal_dir = output_dir / "internal_validation"
    internal_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_path or internal_dir / "processed_input.csv"

    data, audit, source_path, source_mode, provenance, redistribution = _load_input(
        data_path, source_workbook, processed_path
    )
    write_data_quality_report(
        internal_dir / "data_quality_report.md",
        source_path,
        processed_path,
        audit,
        source_mode=source_mode,
        data_provenance=provenance,
        redistribution_status=redistribution,
    )
    values = data["demand"].to_numpy(float)

    predictions = expanding_window_evaluate(
        values,
        models=model_names(),
        initial_train_size=INITIAL_TRAIN_SIZE,
        max_horizon=FORECAST_HORIZON,
        seasonal_period=SEASONAL_PERIOD,
    )
    if not predictions["leakage_check"].all():
        raise AssertionError("Leakage gate failed in stored predictions.")
    comparison = model_comparison(predictions)
    primary_model = select_primary_model(comparison)
    secondary_model = (
        "SEASONAL_NAIVE"
        if primary_model != "SEASONAL_NAIVE"
        else str(comparison.loc[comparison["Model"] != primary_model, "Model"].iloc[0])
    )

    final_points = forecast(primary_model, values, FORECAST_HORIZON, SEASONAL_PERIOD)
    future_periods = np.arange(len(values) + 1, len(values) + FORECAST_HORIZON + 1, dtype=int)
    final_frame = pd.DataFrame(
        {
            "future_period": future_periods,
            "period_label": [period_label(int(period)) for period in future_periods],
            "quarter": ((future_periods - 1) % SEASONAL_PERIOD) + 1,
            "model": primary_model,
            "point_forecast": final_points,
        }
    )

    horizon_errors = error_by_horizon(predictions, primary_model)
    quarter_errors = error_by_quarter(predictions, primary_model)
    selected_predictions = predictions[predictions["model"] == primary_model].copy()
    median_actual = float(selected_predictions["actual"].median())
    selected_predictions["demand_level"] = np.where(
        selected_predictions["actual"] > median_actual,
        "Above pooled median",
        "At/below pooled median",
    )
    level_errors = (
        selected_predictions.groupby("demand_level", as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            Mean_Error=("error", "mean"),
            Count=("error", "size"),
        )
        .sort_values("demand_level")
    )

    robustness_later = model_comparison(
        expanding_window_evaluate(
            values,
            initial_train_size=INITIAL_TRAIN_SIZE + 1,
            max_horizon=FORECAST_HORIZON,
            seasonal_period=SEASONAL_PERIOD,
        )
    )
    robustness_one_step = model_comparison(
        expanding_window_evaluate(
            values,
            initial_train_size=INITIAL_TRAIN_SIZE,
            max_horizon=1,
            seasonal_period=SEASONAL_PERIOD,
        )
    )

    reconciliation_status = "NOT_APPLICABLE_FOR_CSV_INPUT"
    if source_workbook is not None:
        reconciliation = _reconcile_historical_methods(values, source_workbook)
        _write_csv(reconciliation, internal_dir / "historical_reconciliation.csv")
        reconciliation_status = "PASS"

    _write_csv(predictions, output_dir / "backtest_predictions.csv")
    _write_csv(comparison.drop(columns=["_complexity_order"]), output_dir / "model_comparison.csv")
    _write_csv(horizon_errors, output_dir / "error_by_horizon.csv")
    _write_csv(quarter_errors, output_dir / "error_by_quarter.csv")
    _write_csv(level_errors, output_dir / "error_by_demand_level.csv")
    _write_csv(final_frame, output_dir / "final_point_forecast.csv")
    _write_csv(
        robustness_later.drop(columns=["_complexity_order"]),
        output_dir / "robustness_initial_train_9.csv",
    )
    _write_csv(
        robustness_one_step.drop(columns=["_complexity_order"]),
        output_dir / "robustness_one_step.csv",
    )

    candidate_figure_paths = [
        output_dir / "01_historical_time_series.png",
        output_dir / "02_seasonal_pattern.png",
        output_dir / "03_rolling_origin_design.png",
        output_dir / "04_model_performance.png",
        output_dir / "05_actual_vs_backtest.png",
        output_dir / "06_final_point_forecast.png",
    ]
    supplementary_figure_path = output_dir / "07_error_by_horizon.png"
    historical_time_series(data, candidate_figure_paths[0])
    seasonal_pattern(data, candidate_figure_paths[1])
    rolling_origin_design(
        len(data),
        list(range(INITIAL_TRAIN_SIZE, len(data))),
        FORECAST_HORIZON,
        candidate_figure_paths[2],
    )
    model_performance(comparison, candidate_figure_paths[3])
    actual_vs_backtest(data, predictions, primary_model, candidate_figure_paths[4])
    final_point_forecast(data, final_frame, primary_model, candidate_figure_paths[5])
    error_horizon(horizon_errors, primary_model, supplementary_figure_path)

    primary_metrics = comparison.loc[comparison["Model"] == primary_model].iloc[0]
    seasonal_naive_metrics = comparison.loc[comparison["Model"] == "SEASONAL_NAIVE"].iloc[0]
    summary = {
        "analysis": "SEASONAL_DEMAND_FORECASTING_ANALYTICAL_DEMONSTRATION",
        "data_provenance": provenance,
        "redistribution_status": redistribution,
        "source_mode": source_mode,
        "source_file": source_path.name,
        "source_sha256": file_sha256(source_path),
        "observations": int(len(data)),
        "frequency": "Quarterly generic sequential periods",
        "seasonal_period": SEASONAL_PERIOD,
        "complete_seasonal_cycles": len(data) / SEASONAL_PERIOD,
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_horizon_basis": "Fixed four-quarter demonstration horizon established before evaluation.",
        "evaluation": {
            "method": "EXPANDING_WINDOW / ROLLING_ORIGIN",
            "initial_train_size": INITIAL_TRAIN_SIZE,
            "origins": sorted(int(value) for value in predictions["origin_period"].unique()),
            "maximum_horizon": FORECAST_HORIZON,
            "available_horizons_by_origin": [4, 3, 2, 1],
            "forecast_pairs_per_model": int(len(predictions) / predictions["model"].nunique()),
            "random_split_used": False,
            "leakage_check": "PASS",
        },
        "models_evaluated": model_names(),
        "metrics": ["MAE", "RMSE", "MASE", "sMAPE", "Mean Error (diagnostic)"],
        "selected_demonstration_model": primary_model,
        "secondary_reference_model": secondary_model,
        "primary_metrics": {
            "MAE": float(primary_metrics["MAE"]),
            "RMSE": float(primary_metrics["RMSE"]),
            "MASE": float(primary_metrics["MASE"]),
            "sMAPE": float(primary_metrics["sMAPE"]),
            "Mean_Error": float(primary_metrics["Mean_Error"]),
        },
        "seasonal_naive_metrics": {
            "MAE": float(seasonal_naive_metrics["MAE"]),
            "RMSE": float(seasonal_naive_metrics["RMSE"]),
            "MASE": float(seasonal_naive_metrics["MASE"]),
            "sMAPE": float(seasonal_naive_metrics["sMAPE"]),
            "Mean_Error": float(seasonal_naive_metrics["Mean_Error"]),
        },
        "final_point_forecast": final_frame.to_dict(orient="records"),
        "uncertainty": {
            "public_intervals_generated": False,
            "validation_decision": "NOT_DEFENSIBLE_FOR_PUBLICATION",
            "reason": "Ten pooled residual rows with four unique errors cannot support calibrated coverage.",
        },
        "robustness": {
            "later_start_primary": str(robustness_later.iloc[0]["Model"]),
            "one_step_primary": str(robustness_one_step.iloc[0]["Model"]),
            "metric_winners": {
                metric: str(comparison.sort_values(metric).iloc[0]["Model"])
                for metric in ("MAE", "RMSE", "MASE")
            },
        },
        "eda": _eda_summary(data),
        "candidate_figures_pending_rights_review": [path.name for path in candidate_figure_paths],
        "supplementary_figures": [supplementary_figure_path.name],
        "historical_reconciliation": reconciliation_status,
        "small_sample_limitation": (
            "Twelve observations provide only three cycles; rankings have limited stability."
        ),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--data",
        type=Path,
        help="Compatible CSV with period_index, quarter, and demand columns.",
    )
    inputs.add_argument(
        "--source-workbook",
        type=Path,
        help="Explicitly authorized internal workbook for historical reproduction only.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--processed-data",
        type=Path,
        default=None,
        help="Optional execution-copy path; defaults under the ignored local .private_outputs/internal_validation directory.",
    )
    arguments = parser.parse_args()
    if arguments.data is None and arguments.source_workbook is None:
        parser.error(
            "No input supplied. Provide --data path\\to\\quarterly_data.csv "
            "(see data/README.md) or an explicitly authorized --source-workbook path."
        )
    return arguments


if __name__ == "__main__":
    args = parse_args()
    try:
        result = run(
            data_path=args.data,
            source_workbook=args.source_workbook,
            output_dir=args.output_dir,
            processed_path=args.processed_data,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Analysis stopped: {error}") from error
    print(json.dumps(result, indent=2, ensure_ascii=False))
