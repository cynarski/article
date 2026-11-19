"""Regression tests for future-state targets and causal validation."""
import numpy as np
import pandas as pd
import pytest
from lppls_research.event_benchmarks import (
    future_onset_labels, bsadf_asof_features, purged_year_split,
    select_alarm_threshold,
)


def test_future_onset_excludes_active_states_and_censors_tail():
    dates=pd.date_range("2023-01-01",periods=7)
    state=pd.Series([0,0,0,1,1,0,0])
    d=future_onset_labels(pd.Series(dates),state,2)
    assert d.onset.iloc[1]==1
    assert d.onset.iloc[2]==1
    assert pd.isna(d.onset.iloc[3])
    assert pd.isna(d.onset.iloc[-1])
    assert d.label_end.iloc[1]==dates[3]


def test_missing_future_state_is_not_negative():
    dates=pd.date_range("2023-01-01",periods=6)
    state=pd.Series([0,0,np.nan,1,0,0])
    d=future_onset_labels(pd.Series(dates),state,2)
    assert pd.isna(d.onset.iloc[0])
    assert pd.isna(d.onset.iloc[1])


def test_horizon_purge_protects_both_fold_boundaries():
    d=pd.DataFrame({
      "date":pd.to_datetime(["2018-12-01","2018-12-20",
                             "2019-03-01","2019-12-25","2020-02-02"]),
      "label_end":pd.to_datetime(["2018-12-31","2019-02-10",
                                  "2019-03-31","2020-01-10","2020-04-02"])})
    tr,va,te=purged_year_split(d,2020)
    assert len(tr)==1
    assert len(va)==1
    assert len(te)==1


def test_future_reference_is_rejected():
    d=pd.DataFrame({
      "date":["2020-01-01"],"reference_available_at":["2020-01-02"],
      "bsadf_score_last":[1.],"bsadf_critical_last":[.5],
      "gsadf_evaluated_today":[1]})
    with pytest.raises(ValueError,match="future"):
        bsadf_asof_features(d)


def test_no_feasible_threshold_is_not_optimized_on_test():
    threshold,status=select_alarm_threshold(
        [0,1,0,1],[.4,.4,.4,.4],thresholds=[.1,.2],
        max_time_in_alarm=.3)
    assert np.isinf(threshold)
    assert status=="no_feasible_threshold"
