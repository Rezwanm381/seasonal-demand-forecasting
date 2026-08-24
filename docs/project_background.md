# Project Background

This project is a portfolio rebuild derived from graduate forecasting coursework. The historical assignment used a short quarterly demand series and spreadsheet forecasting formulas. The rebuild preserves that analytical lineage while replacing workbook-centric evidence with modular Python, explicit baselines, leakage-safe rolling-origin validation, automated tests, and reproducible figures.

## What the project demonstrates

- recognizing quarterly seasonality and the consequences of limited history;
- establishing naive and seasonal-naive benchmarks before adding structure;
- selecting models with expanding-window evidence rather than a random split;
- checking temporal leakage directly;
- separating backtest evidence from a future point forecast;
- making a conservative uncertainty decision when calibration evidence is insufficient;
- designing a public-safe input interface while data rights remain unresolved.

## Interview discussion prompts

1. Why a seasonal-naive benchmark is essential for seasonal forecasting.
2. Why four rolling origins can be methodologically valid yet statistically fragile.
3. Why the selected seasonal-mean model can be preferable to a slightly more expressive trend model.
4. How future-value perturbation tests provide stronger leakage evidence than code inspection alone.
5. Why removing unsupported prediction intervals improves technical credibility.
6. How the package could evolve once a licensed public dataset or additional seasonal cycles are available.

No claim is made about production deployment, company validation, financial savings, or commercial use.
