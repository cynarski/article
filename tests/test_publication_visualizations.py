"""Regression checks for chronological publication plots."""
import numpy as np
import pandas as pd
import pytest
from scripts.plot_oos_publication_figures import (
    matrix_counts, check_predictions, drawdown_events, onsets
)

def test_matrix_counts_sum_to_full_test():
    """Count all test examples, including both classes."""
    assert matrix_counts([0,0,1,1,1],[0,1,0,1,1]) == (1,1,1,2)

def test_duplicate_and_missing_test_dates_rejected():
    """Do not silently compare different days or incomplete date sequences."""
    base=pd.DataFrame({
        "date":["2024-01-01","2024-01-03"],
        "model":["XGBoost"]*2,"feature_group":["Price-only"]*2,
        "target":[0,1],"alarm":[0,1]})
    with pytest.raises(ValueError,match="Non-continuous"):
        check_predictions(base)

def test_first_alarm_is_left_censored():
    """A positive state at the first OOS date is not a new onset."""
    d=pd.DataFrame({
        "date":pd.date_range("2024-01-01",periods=18),
        "alarm":[1]+[0]*9+[1]+[0]*7})
    assert list(onsets(d)) == [pd.Timestamp("2024-01-11")]

def test_drawdown_peak_excludes_current_price():
    """The rolling reference peak must be known before the event day."""
    prices=pd.DataFrame({"date":pd.date_range("2024-01-01",periods=185),
                         "price":[100.]*180+[110.,109.,94.,94.,94.]})
    dates=drawdown_events(prices,threshold=.1,lookback=180,separation=60)
    assert dates.tolist() == [pd.Timestamp("2024-07-01")]
