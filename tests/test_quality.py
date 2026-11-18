"""Additional checks for input lineage and feature mismatch detection."""
import pandas as pd
import pytest
from lppls_research.quality import normalize_prices,ensure_same_daily_features

def test_market_calendar_permits_weekends_without_interpolation():
    """Market dates may be irregular unless explicitly required to be daily."""
    data=pd.DataFrame({"Date":["2024-01-05","2024-01-08"],"Close":[100.0,101.0]})
    assert len(normalize_prices(data))==2
    with pytest.raises(ValueError,match="nonconsecutive"):
        normalize_prices(data,require_daily=True)

def test_cross_pipeline_dates_must_match():
    """Different date sets must never become a silent inner join."""
    left=pd.DataFrame({"date":["2024-01-01"],"lppls_ci":[0.0]})
    right=pd.DataFrame({"date":["2024-01-02"],"lppls_ci":[0.0]})
    with pytest.raises(ValueError,match="different date sets"):
        ensure_same_daily_features(left,right,columns=["lppls_ci"])

def test_missing_parameter_features_agree():
    """Two matching missing parameter summaries are treated as equal."""
    data=pd.DataFrame({"date":["2024-01-01"],"lppls_m":[float("nan")]})
    assert ensure_same_daily_features(data,data,columns=["lppls_m"])["mismatches"]==0
