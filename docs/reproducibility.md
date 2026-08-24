# Reproducibility

## Environment

Use Python 3.11 or newer and install the small dependency set:

```powershell
python -m pip install -r requirements.txt
```

## Input contract

Supply a local CSV containing exactly these columns:

```text
period_index,quarter,demand
```

For this fixed demonstration, the file must contain 12 numeric rows, periods 1–12 in order, quarter values 1–4 matching the period sequence, and non-negative demand. `data/example_schema.csv` provides only the header and contains no historical values.

## Run

From the repository root:

```powershell
python run_analysis.py --data C:\path\to\authorized_input.csv
```

The command writes all input-derived tables and figures to the ignored local `.private_outputs` directory. It writes the processed input and data-quality details under `.private_outputs/internal_validation`; these artifacts inherit the input's rights and must remain private unless separately cleared.

For a locally authorized copy of the historical workbook:

```powershell
python run_analysis.py --source-workbook C:\path\to\authorized_workbook.xlsx
```

Running without either input option stops with a clear usage error; there is no embedded private path or bundled historical dataset.

## Tests

The test suite uses the Python standard library test runner:

```powershell
python -m unittest discover -s tests -v
```

## Notebooks

The notebooks are optional explanatory views over reusable modules and locally generated outputs. Jupyter and IPython are optional notebook-only dependencies and are not required by the core runner. Run the command-line analysis first, then execute `notebooks/01_exploration.ipynb` and `notebooks/02_forecasting_analysis.ipynb` from top to bottom. No core model or evaluation logic depends on notebook state.

## Validated internal case boundary

With the authorized historical input, the analysis evaluates five models across four expanding-window origins and 10 forecast-target pairs per model; `SEASONAL_MEAN` is selected within that limited demonstration. Exact course-series metrics and point forecasts remain private pending derivative-rights clearance. Reproducing them does not grant permission to redistribute either the historical observations or their derived artifacts. See `LICENSE_STATUS.md` and `data/README.md`.
