"""Dependency-light publication figures built with Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1800, 1000
NAVY = "#17324D"
BLUE = "#2F6B9A"
LIGHT_BLUE = "#DCEAF4"
ORANGE = "#D9772A"
LIGHT_ORANGE = "#F8E4D5"
GREEN = "#3F7D65"
RED = "#B64B4B"
GRAY = "#67717A"
LIGHT_GRAY = "#E7EBEE"
DARK = "#1D2730"
WHITE = "#FFFFFF"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 18), fill=BLUE)
    draw.text((90, 54), title, fill=NAVY, font=_font(42, bold=True))
    draw.text((92, 112), subtitle, fill=GRAY, font=_font(23))
    return image, draw


def _footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.line((90, HEIGHT - 72, WIDTH - 90, HEIGHT - 72), fill=LIGHT_GRAY, width=2)
    draw.text((90, HEIGHT - 53), text, fill=GRAY, font=_font(18))


def _save(image: Image.Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def _scale(values: Iterable[float], low_pad: float = 0.10, high_pad: float = 0.12) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    lower, upper = float(np.nanmin(array)), float(np.nanmax(array))
    span = upper - lower or max(abs(upper), 1.0)
    return max(0.0, lower - low_pad * span), upper + high_pad * span


def _axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_labels: list[str],
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str,
    y_ticks: int = 5,
) -> tuple[callable, callable]:
    left, top, right, bottom = bounds
    draw.line((left, top, left, bottom), fill=DARK, width=3)
    draw.line((left, bottom, right, bottom), fill=DARK, width=3)
    for tick in range(y_ticks + 1):
        fraction = tick / y_ticks
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((left, int(y), right, int(y)), fill=LIGHT_GRAY, width=2)
        draw.text((left - 20, int(y)), f"{value:.0f}", fill=GRAY, font=_font(19), anchor="rm")

    count = len(x_labels)
    x_font = _font(16 if count > 12 else 18)
    for index, label in enumerate(x_labels):
        x = left + (index / max(count - 1, 1)) * (right - left)
        draw.line((int(x), bottom, int(x), bottom + 8), fill=DARK, width=2)
        draw.text((int(x), bottom + 18), label, fill=GRAY, font=x_font, anchor="ma")

    draw.text(((left + right) // 2, bottom + 72), x_label, fill=DARK, font=_font(21, bold=True), anchor="ma")
    draw.text((left, top - 28), y_label, fill=DARK, font=_font(19, bold=True), anchor="ls")

    def map_x(position: float) -> float:
        return left + (position / max(count - 1, 1)) * (right - left)

    def map_y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    return map_x, map_y


def _polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int = 6,
    markers: bool = True,
    dash: bool = False,
) -> None:
    if dash:
        for start, end in zip(points[:-1], points[1:]):
            x1, y1 = start
            x2, y2 = end
            distance = max(int(np.hypot(x2 - x1, y2 - y1)), 1)
            for offset in range(0, distance, 24):
                end_offset = min(offset + 13, distance)
                a = offset / distance
                b = end_offset / distance
                draw.line(
                    (x1 + a * (x2 - x1), y1 + a * (y2 - y1), x1 + b * (x2 - x1), y1 + b * (y2 - y1)),
                    fill=color,
                    width=width,
                )
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    if markers:
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=WHITE, outline=color, width=4)


def historical_time_series(data: pd.DataFrame, output_path: str | Path) -> None:
    image, draw = _canvas(
        "Historical quarterly demand",
        "Twelve observations shown on a generic sequential-quarter index",
    )
    values = data["demand"].to_numpy(float)
    y_min, y_max = _scale(values)
    labels = data["period_label"].tolist()
    map_x, map_y = _axes(
        draw,
        (160, 215, 1700, 800),
        labels,
        y_min,
        y_max,
        "Demand (units unknown)",
        "Generic quarter",
    )
    points = [(map_x(i), map_y(value)) for i, value in enumerate(values)]
    _polyline(draw, points, BLUE)
    for (x, y), value in zip(points, values):
        draw.text((x, y - 18), f"{value:.0f}", fill=NAVY, font=_font(18, bold=True), anchor="ms")
    _footer(draw, "The analysis does not infer calendar dates, target units, or business context from the input series.")
    _save(image, output_path)


def seasonal_pattern(data: pd.DataFrame, output_path: str | Path) -> None:
    image, draw = _canvas(
        "Seasonal pattern by observed year",
        "Quarter-to-quarter shape is visible, but only three cycles are available",
    )
    y_min, y_max = _scale(data["demand"])
    map_x, map_y = _axes(
        draw,
        (180, 220, 1410, 790),
        ["Q1", "Q2", "Q3", "Q4"],
        y_min,
        y_max,
        "Demand (units unknown)",
        "Quarter within cycle",
    )
    colors = [BLUE, ORANGE, GREEN]
    for year, color in zip(sorted(data["year"].unique()), colors):
        frame = data[data["year"] == year].sort_values("quarter")
        points = [(map_x(i), map_y(value)) for i, value in enumerate(frame["demand"].to_numpy(float))]
        _polyline(draw, points, color, width=5)
        draw.rounded_rectangle((1470, 270 + 70 * (year - 1), 1690, 325 + 70 * (year - 1)), radius=12, fill=color)
        draw.text((1580, 297 + 70 * (year - 1)), f"Year {year}", fill=WHITE, font=_font(22, bold=True), anchor="mm")
    quarter_means = data.groupby("quarter")["demand"].mean().to_numpy(float)
    mean_points = [(map_x(i), map_y(value)) for i, value in enumerate(quarter_means)]
    _polyline(draw, mean_points, NAVY, width=7, markers=False, dash=True)
    draw.text((1580, 515), "Quarter mean", fill=NAVY, font=_font(23, bold=True), anchor="mm")
    draw.line((1480, 550, 1680, 550), fill=NAVY, width=7)
    _footer(draw, "Q3 is highest in all three observed cycles; three repetitions are insufficient to claim stable long-run seasonality.")
    _save(image, output_path)


def rolling_origin_design(
    observation_count: int,
    origins: list[int],
    max_horizon: int,
    output_path: str | Path,
) -> None:
    image, draw = _canvas(
        "Expanding-window rolling-origin design",
        "Each origin uses only earlier periods; the available future horizon contracts near the series end",
    )
    left, top = 285, 280
    cell_w, cell_h = 105, 92
    for period in range(1, observation_count + 1):
        draw.text((left + (period - 1) * cell_w + cell_w / 2, top - 45), str(period), fill=DARK, font=_font(20, bold=True), anchor="mm")
    for row, origin in enumerate(origins):
        y = top + row * (cell_h + 24)
        draw.text((left - 35, y + cell_h / 2), f"Origin {origin}", fill=NAVY, font=_font(23, bold=True), anchor="rm")
        available_horizon = min(max_horizon, observation_count - origin)
        for period in range(1, observation_count + 1):
            x = left + (period - 1) * cell_w
            if period <= origin:
                fill, label = LIGHT_BLUE, "T"
            elif period <= origin + available_horizon:
                fill, label = LIGHT_ORANGE, f"h{period - origin}"
            else:
                fill, label = "#F6F7F8", ""
            draw.rounded_rectangle((x + 4, y, x + cell_w - 4, y + cell_h), radius=10, fill=fill, outline=WHITE, width=3)
            draw.text((x + cell_w / 2, y + cell_h / 2), label, fill=DARK, font=_font(20, bold=True), anchor="mm")
    legend_y = 795
    draw.rounded_rectangle((500, legend_y, 570, legend_y + 44), radius=8, fill=LIGHT_BLUE)
    draw.text((590, legend_y + 22), "Training history", fill=DARK, font=_font(21), anchor="lm")
    draw.rounded_rectangle((880, legend_y, 950, legend_y + 44), radius=8, fill=LIGHT_ORANGE)
    draw.text((970, legend_y + 22), "Forecast target", fill=DARK, font=_font(21), anchor="lm")
    _footer(draw, "Automated gate at every origin: max(training_timestamp) < min(forecast_target_timestamp).")
    _save(image, output_path)


def model_performance(comparison: pd.DataFrame, output_path: str | Path) -> None:
    frame = comparison.sort_values("MASE", ascending=True).reset_index(drop=True)
    image, draw = _canvas(
        "Rolling-origin model comparison",
        "Each error uses its origin's training-only seasonal scale; lower MASE is better",
    )
    left, top, right, bottom = 880, 240, 1640, 790
    max_value = max(1.15, float(frame["MASE"].max()) * 1.12)
    row_h = (bottom - top) / len(frame)
    reference_x = left + (1.0 / max_value) * (right - left)
    draw.line((reference_x, top - 20, reference_x, bottom + 10), fill=ORANGE, width=4)
    draw.text((reference_x, top - 35), "MASE = 1.0", fill=ORANGE, font=_font(19, bold=True), anchor="ms")
    for index, row in frame.iterrows():
        y = top + index * row_h
        bar_y1, bar_y2 = y + 22, y + row_h - 20
        bar_right = left + float(row["MASE"]) / max_value * (right - left)
        color = BLUE if index == 0 else LIGHT_BLUE
        draw.text((left - 30, (bar_y1 + bar_y2) / 2), str(row["Model"]), fill=NAVY, font=_font(18, bold=index == 0), anchor="rm")
        draw.rounded_rectangle((left, bar_y1, bar_right, bar_y2), radius=12, fill=color)
        draw.text((bar_right + 18, (bar_y1 + bar_y2) / 2), f"{row['MASE']:.3f}", fill=DARK, font=_font(21, bold=True), anchor="lm")
        draw.text((120, (bar_y1 + bar_y2) / 2), f"MAE {row['MAE']:.2f}  |  RMSE {row['RMSE']:.2f}", fill=GRAY, font=_font(18), anchor="lm")
    _footer(draw, "Metrics pool 10 forecasts from four common expanding origins; rankings remain statistically fragile.")
    _save(image, output_path)


def actual_vs_backtest(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
) -> None:
    one_step = predictions[(predictions["model"] == model_name) & (predictions["horizon"] == 1)].copy()
    actual = data["demand"].to_numpy(float)
    all_values = np.concatenate([actual, one_step["prediction"].to_numpy(float)])
    y_min, y_max = _scale(all_values)
    image, draw = _canvas(
        "Actual versus one-step rolling-origin forecasts",
        f"{model_name}: each displayed forecast was created before its target observation was available",
    )
    labels = data["period_label"].tolist()
    map_x, map_y = _axes(draw, (160, 220, 1680, 790), labels, y_min, y_max, "Demand (units unknown)", "Generic quarter")
    actual_points = [(map_x(i), map_y(value)) for i, value in enumerate(actual)]
    _polyline(draw, actual_points, GRAY, width=5)
    forecast_points = [
        (map_x(int(row.target_period) - 1), map_y(float(row.prediction)))
        for row in one_step.itertuples()
    ]
    _polyline(draw, forecast_points, ORANGE, width=6)
    draw.line((1180, 180, 1240, 180), fill=GRAY, width=5)
    draw.text((1260, 180), "Actual", fill=DARK, font=_font(20), anchor="lm")
    draw.line((1420, 180, 1480, 180), fill=ORANGE, width=6)
    draw.text((1500, 180), "One-step forecast", fill=DARK, font=_font(20), anchor="lm")
    _footer(draw, "Backtest performance is distinct from the future forecast for periods 13-16.")
    _save(image, output_path)


def final_point_forecast(
    data: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    model_name: str,
    output_path: str | Path,
) -> None:
    """Plot point forecasts only; no nominal interval coverage is asserted."""
    actual = data["demand"].to_numpy(float)
    point = forecast_frame["point_forecast"].to_numpy(float)
    all_values = np.concatenate([actual, point])
    y_min, y_max = _scale(all_values, low_pad=0.08, high_pad=0.10)
    total_periods = len(actual) + len(point)
    labels = [
        f"P{i}" if (i % 2 == 1 or i >= len(actual) - 1) else ""
        for i in range(1, total_periods + 1)
    ]
    image, draw = _canvas(
        "Final four-quarter demonstration forecast",
        f"{model_name} fitted to 12 observations; point forecasts only",
    )
    map_x, map_y = _axes(draw, (160, 220, 1680, 790), labels, y_min, y_max, "Demand (units unknown)", "Generic sequential period")
    future_positions = list(range(len(actual), total_periods))
    _polyline(draw, [(map_x(i), map_y(v)) for i, v in enumerate(actual)], GRAY, width=5)
    bridge = [(map_x(len(actual) - 1), map_y(actual[-1]))] + [
        (map_x(i), map_y(v)) for i, v in zip(future_positions, point)
    ]
    _polyline(draw, bridge, BLUE, width=7)
    divider_x = (map_x(len(actual) - 1) + map_x(len(actual))) / 2
    draw.line((divider_x, 220, divider_x, 790), fill=ORANGE, width=4)
    draw.text((divider_x + 16, 245), "Forecast begins", fill=ORANGE, font=_font(20, bold=True), anchor="la")
    for index, value in zip(future_positions, point):
        draw.text((map_x(index), map_y(value) - 17), f"{value:.1f}", fill=NAVY, font=_font(19, bold=True), anchor="ms")
    draw.line((1180, 180, 1240, 180), fill=GRAY, width=5)
    draw.text((1260, 180), "Observed", fill=DARK, font=_font(20), anchor="lm")
    draw.line((1420, 180, 1480, 180), fill=BLUE, width=7)
    draw.text((1500, 180), "Point forecast", fill=DARK, font=_font(20), anchor="lm")
    _footer(draw, "Point forecasts only; validation rejected the earlier residual bands for public interval interpretation.")
    _save(image, output_path)


def error_horizon(error_frame: pd.DataFrame, model_name: str, output_path: str | Path) -> None:
    image, draw = _canvas(
        "Forecast error by horizon",
        f"{model_name}: horizon-specific MAE with rapidly declining sample counts",
    )
    left, top, right, bottom = 260, 240, 1600, 790
    max_value = max(float(error_frame["MAE"].max()) * 1.18, 1.0)
    count = len(error_frame)
    slot = (right - left) / count
    for tick in range(6):
        value = tick / 5 * max_value
        y = bottom - tick / 5 * (bottom - top)
        draw.line((left, y, right, y), fill=LIGHT_GRAY, width=2)
        draw.text((left - 25, y), f"{value:.0f}", fill=GRAY, font=_font(19), anchor="rm")
    draw.line((left, top, left, bottom), fill=DARK, width=3)
    draw.line((left, bottom, right, bottom), fill=DARK, width=3)
    for index, row in error_frame.reset_index(drop=True).iterrows():
        center = left + (index + 0.5) * slot
        height = float(row["MAE"]) / max_value * (bottom - top)
        draw.rounded_rectangle((center - 90, bottom - height, center + 90, bottom), radius=14, fill=BLUE)
        draw.text((center, bottom - height - 18), f"MAE {row['MAE']:.2f}", fill=NAVY, font=_font(21, bold=True), anchor="ms")
        draw.text((center, bottom + 25), f"Horizon {int(row['horizon'])}", fill=DARK, font=_font(21, bold=True), anchor="ma")
        draw.text((center, bottom + 62), f"n = {int(row['Count'])}", fill=GRAY, font=_font(19), anchor="ma")
    draw.text((left, top - 35), "Mean absolute error", fill=DARK, font=_font(19, bold=True), anchor="ls")
    _footer(draw, "Horizon 4 has only one error; this diagnostic describes the demonstration and cannot establish a stable horizon effect.")
    _save(image, output_path)
