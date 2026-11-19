"""Regression tests for historically reproducible sampled-BSADF Monte Carlo."""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from bubble_tools import (monte_carlo_bsadf_reference, _prefixes, _adf_t_for_windows)


def test_adf_matches_independent_statsmodels():
    """Validate the lag-zero, constant-only ADF coefficient t-statistic."""
    rng = np.random.default_rng(123)
    x = np.cumsum(rng.normal(size=400))
    result = _adf_t_for_windows(_prefixes(x), np.array([44]), 300)[0]
    expected = adfuller(x[44:301], maxlag=0, autolag=None, regression="c")[0]
    assert abs(result-expected) < 1e-8


def test_monte_carlo_critical_value_does_not_use_final_dataset_length():
    """A fixed historical checkpoint must have the same reference cutoff."""
    rng = np.random.default_rng(11)
    prices = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=410),
        "price": np.exp(.05 + np.cumsum(rng.normal(scale=.01, size=410))),
    })
    full, _ = monte_carlo_bsadf_reference(
        prices, min_window=90, endpoint_step=7, start_step=14, reps=39, seed=42
    )
    partial, _ = monte_carlo_bsadf_reference(
        prices.iloc[:260].copy(), min_window=90, endpoint_step=7,
        start_step=14, reps=39, seed=42
    )
    check = partial.loc[
        partial.gsadf_evaluated_today.eq(1), ["date", "bsadf_critical_last"]
    ]
    paired = check.merge(
        full[["date", "bsadf_critical_last"]],
        on="date", suffixes=("_past", "_full")
    )
    scheduled = ((paired.date - prices.date.iloc[89]).dt.days % 7 == 0)
    assert np.allclose(
        paired.loc[scheduled, "bsadf_critical_last_past"],
        paired.loc[scheduled, "bsadf_critical_last_full"],
        atol=1e-8, rtol=0
    )
