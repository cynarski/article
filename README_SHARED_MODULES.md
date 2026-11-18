# Reusable rolling-fit features and temporal guards

This first-stage refactor adds the installable, market-agnostic
`lppls_research` package. Its public API does not depend on notebooks, BTC,
Yahoo Finance, XGBoost, or local CSV paths.

## Public API

- `RollingFitAggregator` aggregates any accepted/rejected rolling-window fits
  into daily counts, accepted fractions and optional numeric summaries.
- `LPPLSAcceptanceCriteria` applies configurable LPPLS acceptance conditions.
- `build_lppls_daily_features` reproduces the legacy daily LPPLS schema.
- `TemporalAvailabilityValidator` checks rolling-window order, prediction
  calendars, feature/reference as-of timestamps, future-label availability
  at fold boundaries and cross-pipeline daily count equivalence.

All public docstrings are in English. Failed fits count toward the
Confidence Indicator denominator, and dates without fits remain in the output
when an explicit calendar is supplied. Trading-day calendars need not be
consecutive; set `require_consecutive_days=True` only for daily markets.

## Install / test

```bash
pip install -e ".[test]"
pytest -q tests
```

The two notebooks insert `src` on `sys.path`, so editable installation is
optional in the existing research repository.

## A generic use case (no LPPLS dependency)

```python
import pandas as pd
from lppls_research import RollingFitAggregator, TemporalAvailabilityValidator

fits = pd.DataFrame({
    "as_of": ["2025-01-02", "2025-01-02"],
    "passed": [True, False],
    "score": [0.8, 0.2],
})
features = RollingFitAggregator(
    end_col="as_of", accepted_col="passed", prefix="signal",
    metrics={"accepted_score_median": ("score", "median")},
).transform(fits, calendar=pd.Series(pd.date_range("2025-01-01", periods=3)))

TemporalAvailabilityValidator.check_as_of(
    pd.DataFrame({"date": ["2025-01-02"], "available_at": ["2025-01-02"]}),
    available_col="available_at",
)
```

## Project-specific usage

```python
fits = pd.read_csv("lppls_rolling_results.csv")  # all windows
features = build_lppls_daily_features(fits, calendar=prices["date"])

# Ensure that independently prepared data agree on every corresponding date.
TemporalAvailabilityValidator.check_matching_counts(
    old_daily, features,
    mapping={"n_all": "lppls_n_all", "n_accepted": "lppls_n_accepted"},
)

# For future targets, make sure labels end strictly before subsequent folds.
TemporalAvailabilityValidator.check_split(
    train, validation, test, label_end_col="label_end_h60"
)
```

## Scope and reproducibility

This change deliberately **does not** re-run or revise model results.
Current 08b/14 output CSV files and 14/15 publication results must be
regenerated after merging. The checks deliberately fail on inconsistent fit
files rather than silently choosing a similarly named CSV.

The research notebooks 13–16 still import the historical `bubble_tools.py`
module; it is not included in the current GitHub tree. Restore/version this
module as a separate reproducibility task before claiming a clean clone can
run the complete research pipeline.
