"""Guard against misleading combined OOS timelines from incompatible samples."""
import numpy as np
import pandas as pd
import pytest
from scripts.plot_oos_multi_method_timelines import aligned_alarms, contiguous_intervals


def _records():
    dates = list(pd.date_range("2024-01-01", periods=5))
    labels = [0, 1, 1, 0, 0]
    fixed = pd.DataFrame({
        "date": dates * 2, "model": ["StatisticalRule"] * 10,
        "feature_group": ["LPPLS-CI"] * 5 + ["BSADF-gap"] * 5,
        "target": labels * 2, "alarm": labels * 2
    })
    tuned = pd.concat([
        pd.DataFrame({
            "date": dates, "drop_pct": [10] * 5, "horizon": [60] * 5,
            "model": [model] * 5, "feature_group": ["Price+LPPLS"] * 5,
            "y_true": labels, "y_pred": labels
        }) for model in ["LogisticRegression", "DecisionTree", "XGBoost",
                        "LightGBM", "CatBoost"]
    ], ignore_index=True)
    return tuned, fixed


def test_same_date_and_label_across_all_methods():
    """Synthetic five-day input must fail only on frozen sample size."""
    tuned, fixed = _records()
    with pytest.raises(ValueError, match="Expected 1401"):
        aligned_alarms(tuned, fixed)


def test_reject_disagreement_in_future_target():
    """A changed target in even one model must abort figure generation."""
    tuned, fixed = _records()
    tuned.loc[0, "y_true"] = 0
    with pytest.raises(ValueError, match="Inconsistent OOS"):
        aligned_alarms(tuned, fixed)


def test_reject_duplicate_calendar_dates():
    """Never average or silently deduplicate binary alarm observations."""
    tuned, fixed = _records()
    tuned.loc[0, "date"] = tuned.loc[1, "date"]
    with pytest.raises(ValueError, match="Duplicate OOS"):
        aligned_alarms(tuned, fixed)


def test_half_open_alarm_intervals():
    """One positive observation is represented by exactly one calendar day."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=6))
    observed = contiguous_intervals(dates, np.array([0, 1, 1, 0, 1, 0]))
    assert observed == [
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")),
        (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06"))
    ]
