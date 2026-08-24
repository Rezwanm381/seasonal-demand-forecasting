"""Input loading, generic-period reconstruction, and data-quality gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_COLUMNS = ("period_index", "quarter", "demand")
REQUIRED_COLUMNS = ("period_index", "period_label", "year", "quarter", "demand")


@dataclass(frozen=True)
class DataQualityAudit:
    observation_count: int
    chronological_order: bool
    duplicate_periods: list[int]
    missing_periods: list[int]
    missing_target_values: int
    irregular_spacing: bool
    impossible_negative_values: list[float]
    quarter_mapping_consistent: bool
    frequency: str
    seasonal_period: int
    complete_seasonal_cycles: float
    authentic_dates_available: bool
    units: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    """Return a stable SHA-256 checksum without modifying the file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _find_source_header(sheet: pd.DataFrame) -> int:
    for index, row in sheet.iterrows():
        values = [str(value).strip().lower() for value in row.iloc[:3].tolist()]
        if values == ["period", "quarter", "demand"]:
            return int(index)
    raise ValueError("Could not locate the Period / Quarter / Demand header in the source workbook.")


def _standardize_series(
    frame: pd.DataFrame,
    seasonal_period: int,
    expected_observations: int,
) -> pd.DataFrame:
    """Validate the three public input fields and derive generic labels."""
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    candidate = frame.loc[:, INPUT_COLUMNS].copy()
    numeric = {
        column: pd.to_numeric(candidate[column], errors="coerce") for column in INPUT_COLUMNS
    }
    invalid_period = numeric["period_index"].isna() | (numeric["period_index"] % 1 != 0)
    invalid_quarter = numeric["quarter"].isna() | (numeric["quarter"] % 1 != 0)
    invalid_demand = numeric["demand"].isna() | ~np.isfinite(numeric["demand"])
    failures = []
    if invalid_period.any():
        failures.append(f"invalid period_index rows: {(candidate.index[invalid_period] + 2).tolist()}")
    if invalid_quarter.any():
        failures.append(f"invalid quarter rows: {(candidate.index[invalid_quarter] + 2).tolist()}")
    if invalid_demand.any():
        failures.append(f"invalid demand rows: {(candidate.index[invalid_demand] + 2).tolist()}")
    if failures:
        raise ValueError("Input parsing failed: " + "; ".join(failures))

    if len(candidate) != expected_observations:
        raise ValueError(
            "This validated demonstration expects exactly "
            f"{expected_observations} observations; received {len(candidate)}. "
            "A different series length requires a separately validated configuration."
        )

    output = pd.DataFrame(
        {
            "period_index": numeric["period_index"].astype(int),
            "quarter": numeric["quarter"].astype(int),
            "demand": numeric["demand"].astype(float),
        }
    )
    output["year"] = ((output["period_index"] - 1) // seasonal_period + 1).astype(int)
    output["period_label"] = output.apply(
        lambda row: f"Y{int(row['year'])} Q{int(row['quarter'])}", axis=1
    )
    return output[["period_index", "period_label", "year", "quarter", "demand"]].reset_index(
        drop=True
    )


def extract_workbook_series(
    workbook_path: str | Path,
    seasonal_period: int = 4,
    expected_observations: int = 12,
) -> pd.DataFrame:
    """Extract the contiguous historical table from an authorized workbook."""
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(workbook_path)

    sheet = pd.read_excel(
        workbook_path,
        sheet_name="Problem1_NoTrend",
        header=None,
        engine="openpyxl",
    )
    header_row = _find_source_header(sheet)
    rows: list[list[object]] = []
    for _, row in sheet.iloc[header_row + 1 :, :3].iterrows():
        values = row.tolist()
        if pd.isna(values).all():
            break
        if pd.isna(values).any():
            raise ValueError("The historical source table contains a partially blank row.")
        rows.append(values)

    candidate = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    return _standardize_series(candidate, seasonal_period, expected_observations)


def load_csv_series(
    csv_path: str | Path,
    seasonal_period: int = 4,
    expected_observations: int = 12,
) -> pd.DataFrame:
    """Load a compatible CSV without assuming dates, units, or business context."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    return _standardize_series(frame, seasonal_period, expected_observations)


def extract_historical_forecasts(workbook_path: str | Path) -> pd.DataFrame:
    """Read cached forecast cells for internal historical reconciliation only."""
    workbook_path = Path(workbook_path)
    no_trend = pd.read_excel(
        workbook_path,
        sheet_name="Problem1_NoTrend",
        header=None,
        engine="openpyxl",
    )
    with_trend = pd.read_excel(
        workbook_path,
        sheet_name="Problem2_Trend",
        header=None,
        engine="openpyxl",
    )
    return pd.DataFrame(
        {
            "future_period": np.arange(13, 17, dtype=int),
            "quarter": np.arange(1, 5, dtype=int),
            "workbook_seasonal_mean": pd.to_numeric(no_trend.iloc[4:8, 7]).to_numpy(float),
            "workbook_trend_multiplicative": pd.to_numeric(with_trend.iloc[4:8, 12]).to_numpy(float),
        }
    )


def audit_series(data: pd.DataFrame, seasonal_period: int = 4) -> DataQualityAudit:
    """Audit a generic quarterly sequence without silently repairing malformed values."""
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    periods = pd.to_numeric(data["period_index"], errors="coerce")
    quarters = pd.to_numeric(data["quarter"], errors="coerce")
    targets = pd.to_numeric(data["demand"], errors="coerce")
    valid_periods = periods.dropna()
    observed = set(valid_periods.astype(int))
    expected = set(range(1, int(valid_periods.max()) + 1)) if not valid_periods.empty else set()
    duplicate_values = periods[periods.duplicated(keep=False)].dropna().astype(int).unique().tolist()
    invalid_periods = periods.isna() | (periods % 1 != 0)
    invalid_quarters = quarters.isna() | (quarters % 1 != 0)
    invalid_targets = targets.isna() | ~np.isfinite(targets)
    spacing = periods.diff().dropna()
    regular_start = bool(not periods.empty and not invalid_periods.any() and periods.iloc[0] == 1)

    if invalid_periods.any() or invalid_quarters.any():
        quarter_mapping_consistent = False
    else:
        expected_quarter = ((periods.astype(int) - 1) % seasonal_period) + 1
        quarter_mapping_consistent = bool(expected_quarter.equals(quarters.astype(int)))

    return DataQualityAudit(
        observation_count=int(len(data)),
        chronological_order=bool(not invalid_periods.any() and periods.is_monotonic_increasing),
        duplicate_periods=sorted(int(value) for value in duplicate_values),
        missing_periods=sorted(int(value) for value in expected - observed),
        missing_target_values=int(invalid_targets.sum()),
        irregular_spacing=bool(not regular_start or not spacing.eq(1).all()),
        impossible_negative_values=[
            float(value) for value in targets[(~invalid_targets) & (targets < 0)].tolist()
        ],
        quarter_mapping_consistent=quarter_mapping_consistent,
        frequency="Quarterly (generic sequential periods; authentic calendar dates unavailable)",
        seasonal_period=int(seasonal_period),
        complete_seasonal_cycles=float(len(data) / seasonal_period),
        authentic_dates_available=False,
        units="UNKNOWN",
    )


def validate_audit(audit: DataQualityAudit) -> None:
    """Stop the pipeline when a data-quality gate is not satisfied."""
    failures: list[str] = []
    if not audit.chronological_order:
        failures.append("periods are invalid or not chronological")
    if audit.duplicate_periods:
        failures.append(f"duplicate periods: {audit.duplicate_periods}")
    if audit.missing_periods:
        failures.append(f"missing periods: {audit.missing_periods}")
    if audit.missing_target_values:
        failures.append(f"missing or invalid target values: {audit.missing_target_values}")
    if audit.irregular_spacing:
        failures.append("periods do not form a unit-spaced sequence beginning at 1")
    if audit.impossible_negative_values:
        failures.append("negative demand values")
    if not audit.quarter_mapping_consistent:
        failures.append("quarter values do not match the sequential period index")
    if failures:
        raise ValueError("Data quality gate failed: " + "; ".join(failures))


def _write_processed(data: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False, float_format="%.10g")


def prepare_development_data(
    workbook_path: str | Path,
    output_csv: str | Path,
    seasonal_period: int = 4,
    expected_observations: int = 12,
) -> tuple[pd.DataFrame, DataQualityAudit]:
    """Create a validated local copy from an explicitly authorized workbook."""
    data = extract_workbook_series(
        workbook_path, seasonal_period=seasonal_period, expected_observations=expected_observations
    )
    audit = audit_series(data, seasonal_period=seasonal_period)
    validate_audit(audit)
    _write_processed(data, output_csv)
    return data, audit


def prepare_csv_data(
    csv_path: str | Path,
    output_csv: str | Path,
    seasonal_period: int = 4,
    expected_observations: int = 12,
) -> tuple[pd.DataFrame, DataQualityAudit]:
    """Create a validated local copy from a user-supplied compatible CSV."""
    data = load_csv_series(
        csv_path, seasonal_period=seasonal_period, expected_observations=expected_observations
    )
    audit = audit_series(data, seasonal_period=seasonal_period)
    validate_audit(audit)
    _write_processed(data, output_csv)
    return data, audit


def write_data_quality_report(
    report_path: str | Path,
    source_path: str | Path,
    processed_path: str | Path,
    audit: DataQualityAudit,
    *,
    source_mode: str,
    data_provenance: str,
    redistribution_status: str,
) -> None:
    """Write a deterministic audit report beside non-public execution artifacts."""
    status = lambda condition: "PASS" if condition else "FAIL"
    report = f"""# Data quality report

Generated by `src/data_prep.py` from an explicitly supplied input.

## Source and reconstruction

- Input file: `{Path(source_path).name}`
- Input mode: `{source_mode}`
- Input SHA-256: `{file_sha256(source_path)}`
- Processed execution copy filename: `{Path(processed_path).name}`
- Data provenance: `{data_provenance}`
- Redistribution status: `{redistribution_status}`
- Target: `demand`
- Units: `UNKNOWN`
- Time representation: generic sequential quarters; no calendar dates asserted
- Frequency: {audit.frequency}
- Observations: {audit.observation_count}
- Seasonal period: {audit.seasonal_period}
- Complete seasonal cycles: {audit.complete_seasonal_cycles:.1f}

## Programmatic checks

| Check | Result | Detail |
|---|---:|---|
| Chronological ordering | {status(audit.chronological_order)} | `period_index` is a valid monotonic sequence |
| Duplicate periods | {status(not audit.duplicate_periods)} | {audit.duplicate_periods or 'None'} |
| Missing periods | {status(not audit.missing_periods)} | {audit.missing_periods or 'None'} |
| Missing/invalid target values | {status(audit.missing_target_values == 0)} | {audit.missing_target_values} |
| Regular spacing | {status(not audit.irregular_spacing)} | Consecutive integer quarters beginning at 1 |
| Impossible negative demand | {status(not audit.impossible_negative_values)} | {audit.impossible_negative_values or 'None'} |
| Quarter mapping | {status(audit.quarter_mapping_consistent)} | Quarter equals `((period_index - 1) mod 4) + 1` |
| Date parsing | NOT APPLICABLE | Authentic dates are not required or inferred |

## Transformations

No target value is edited, imputed, winsorized, deleted, or reordered. The loader validates the three input fields and derives generic `year` and `period_label` values. Any dates, labels, or units outside the required schema are not treated as verified analytical facts.

## Data gate

`PASS`: period ordering, duplicates, missing periods, target validity, spacing, non-negativity, and quarter mapping satisfy the validated demonstration requirements.

This execution copy inherits the input's rights status and is not automatically approved for redistribution.
"""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
