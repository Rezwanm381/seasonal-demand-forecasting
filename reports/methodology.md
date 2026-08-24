# Methodology

## Scope

This portfolio rebuild demonstrates disciplined seasonal forecasting on a short quarterly series derived from graduate forecasting coursework. It is an analytical demonstration, not an operational deployment or a claim of commercial impact.

## Data and public interface

The validated local analysis uses 12 quarterly observations (three seasonal cycles) with a target labelled `demand`; its units and original external source are not established. The evidence supports `DATA_PROVENANCE = COURSE_PROVIDED` and `REDISTRIBUTION_STATUS = VERIFY_BEFORE_PUBLICATION`.

The intended public export does not contain the historical observations. A user supplies either:

- a CSV with `period_index`, `quarter`, and `demand`; or
- the locally authorized historical workbook through `--source-workbook`.

The input loader enforces numeric values, exactly 12 observations for this demonstration, contiguous periods beginning at 1, chronological order, no duplicate periods, no missing targets, quarterly consistency, and non-negative demand. Repairs are never made silently.

## Forecasting task

- Target: demand, with units unknown.
- Frequency: quarterly.
- Seasonal period: 4.
- Horizon: 4 quarters, fixed before model comparison to represent one complete seasonal cycle.
- Objective: compare compact, interpretable methods under leakage-safe historical simulation.

## Models

Five deliberately parsimonious models are evaluated:

1. `NAIVE`: repeats the most recent observation.
2. `SEASONAL_NAIVE`: repeats the observation from four quarters earlier.
3. `SEASONAL_MEAN`: averages prior observations for each quarter.
4. `TREND_SEASONAL_ADDITIVE`: least-squares trend plus additive quarter effects.
5. `TREND_SEASONAL_MULTIPLICATIVE`: least-squares trend applied to normalized multiplicative quarter factors.

The two trend-seasonal methods reconstruct historical coursework formulas, but their current performance numbers come from the new backtest. No ARIMA, Prophet, tree ensemble, or neural model is added because 12 observations cannot support their extra flexibility credibly.

## Rolling-origin validation

Validation uses an expanding window. The initial training window contains periods 1–8 (two full cycles). Origins occur after periods 8, 9, 10, and 11. Each origin forecasts as many as four steps, truncated at the end of the observed series; therefore the origins contribute 4, 3, 2, and 1 target observations. This yields 10 forecast-target pairs per model and 50 predictions overall.

Every split enforces:

`max(training_period) < min(forecast_target_period)`

The assertion runs before forecasting and stops evaluation on overlap. No random split is used. Metrics are pooled across the 10 comparable target pairs for each model.

## Metrics

- MAE gives an interpretable average absolute miss in the target's unknown units.
- RMSE gives extra weight to larger misses.
- MASE scales each error using an in-origin seasonal-naive training scale, making comparisons meaningful without known units.
- sMAPE is retained as a bounded relative-error diagnostic because the observed targets are positive; it is not used alone for selection.

Model selection considers MAE, RMSE, MASE, rank consistency, interpretability, and complexity. The selected model is described only as the best-performing model within this evaluated demonstration.

## Robustness and error analysis

Two compact sensitivity checks are used: a later initial training window of nine observations and a one-step-ahead expanding-window evaluation. Errors are also summarized by horizon, quarter, and an observed-demand group split at the pooled median. Those summaries are descriptive because each group is small.

## Uncertainty decision

The earlier build explored empirical residual quantile bands. Module 7.5A validation found those bands not defensible for publication: only 10 pooled errors exist per model, and longer-horizon samples are especially sparse. The candidate analysis therefore produces point forecasts only. The historical interval artifact is retained solely in the private archival workspace as audit evidence and must not be presented as validated uncertainty.

## Output separation

All input-derived results, figures, processed series, coursework-formula reconciliation, and rejected interval evidence are written under the ignored local `.private_outputs/` tree. The interim public allowlist includes only a data-independent validation schematic; this boundary preserves a reproducible user-supplied-data workflow without approving restricted or uncleared derivatives.
