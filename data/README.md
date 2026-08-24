# Data interface

The intended public export does not contain the 12 historical coursework values. The archival workspace retains private source and validation material outside that export.

`DATA_PROVENANCE = COURSE_PROVIDED`  
`REDISTRIBUTION_STATUS = VERIFY_BEFORE_PUBLICATION`

The original source, license, dates, units, product identity, collection process, and real-versus-constructed status have not been verified. Historical raw and processed data must therefore remain outside any public release.

## Compatible CSV schema

Provide a CSV with exactly these columns:

| Column | Requirement |
|---|---|
| `period_index` | Integers 1–12 in chronological order, without gaps or duplicates |
| `quarter` | Integers 1–4 matching `((period_index - 1) mod 4) + 1` |
| `demand` | Twelve finite, non-negative numeric values |

`example_schema.csv` contains the header only. It intentionally contains no historical or synthetic observations that could be confused with the validated coursework series.

The analysis derives generic labels such as `Y1 Q1`. It does not interpret other columns as verified dates, units, or business context.

## Run with authorized data

```powershell
python run_analysis.py --data "path\to\authorized_quarterly_data.csv"
```

The user is responsible for ensuring the supplied data may be used and redistributed. A different observation count or seasonal structure requires a separately validated configuration rather than silent truncation or remapping.

Processed execution copies are written under the ignored local `.private_outputs/internal_validation` directory. They inherit the input data's rights status and must not be published automatically.
