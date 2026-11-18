"""Unit tests for generic fit aggregation and temporal information guards."""
import numpy as np
import pandas as pd
import pytest

from lppls_research import (LPPLSAcceptanceCriteria, RollingFitAggregator,
                            TemporalAvailabilityValidator as TV,
                            build_lppls_daily_features)


def sample_fits():
    """Return two accepted/rejected LPPLS fits plus an unsuccessful fit."""
    end = pd.Timestamp("2024-01-02")
    tc = float(end.toordinal())
    return pd.DataFrame({
        "win_start": ["2023-12-01", "2023-12-02", "2023-12-03"],
        "win_end": [end, end, end], "ok": [True, True, False],
        "tc": [tc + 10, tc + 10, np.nan],
        "m": [0.5, 1.5, np.nan], "w": [8, 8, np.nan],
        "B": [-1.0, -1.0, np.nan], "r2": [.95, .95, np.nan],
        "C": [.2, .2, np.nan], "n": [30, 31, 32]
    })


def test_lppls_aggregates_all_denominators_and_preserves_empty_days():
    calendar = pd.date_range("2024-01-01", periods=3)
    result = build_lppls_daily_features(sample_fits(), calendar=calendar)
    assert len(result) == 3
    day = result.loc[result.date.eq(pd.Timestamp("2024-01-02"))].iloc[0]
    assert day.lppls_n_all == 3
    assert day.lppls_n_accepted == 1
    assert day.lppls_ci == pytest.approx(1 / 3)
    assert day.lppls_tgap == pytest.approx(10)
    assert pd.isna(day.lppls_tgap_std)
    empty = result.loc[result.date.eq(pd.Timestamp("2024-01-03"))].iloc[0]
    assert empty.lppls_ci == 0
    assert empty.lppls_n_accepted == 0
    assert pd.isna(empty.lppls_m)


def test_filters_are_configurable():
    fit = sample_fits()
    result = LPPLSAcceptanceCriteria(m_upper=2.0).evaluate(fit)
    assert result.accepted.tolist() == [True, True, False]


def test_generic_aggregator():
    frame = pd.DataFrame({"timestamp": ["2024-01-01"] * 2,
                          "passed": [True, False], "loss": [2., 5.]})
    summary = RollingFitAggregator(end_col="timestamp", accepted_col="passed",
                                   prefix="forecast", metrics={"loss_med": ("loss", "median")}).transform(frame)
    assert summary.forecast_ci.iloc[0] == 0.5
    assert summary.loss_med.iloc[0] == 2


def test_future_feature_rejected():
    sample = pd.DataFrame({"date": ["2024-01-01"], "ready": ["2024-01-02"]})
    with pytest.raises(ValueError, match="Future"):
        TV.check_as_of(sample, available_col="ready")


def test_train_labels_crossing_validation_rejected():
    train = pd.DataFrame({"date": ["2024-01-01"], "label_end": ["2024-01-05"]})
    val = pd.DataFrame({"date": ["2024-01-03"], "label_end": ["2024-01-03"]})
    test = pd.DataFrame({"date": ["2024-01-04"]})
    with pytest.raises(ValueError, match="leaks"):
        TV.check_split(train, val, test, label_end_col="label_end")


def test_valid_asof_current_state_and_irregular_calendar():
    train = pd.DataFrame({"date": ["2024-01-01"], "ready": ["2024-01-01"]})
    val = pd.DataFrame({"date": ["2024-01-03"], "ready": ["2024-01-03"]})
    test = pd.DataFrame({"date": ["2024-01-05"], "ready": ["2024-01-05"]})
    TV.check_split(train, val, test, feature_ready_col="ready")


def test_mismatched_daily_counts_raise():
    base = pd.DataFrame({"date": ["2024-01-02"], "count": [2]})
    actual = pd.DataFrame({"date": ["2024-01-02"], "fit_n_all": [3]})
    with pytest.raises(ValueError, match="count != fit_n_all"):
        TV.check_matching_counts(base, actual, mapping={"count": "fit_n_all"})
