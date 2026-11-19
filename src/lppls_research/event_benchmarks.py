"""Causal future-event labels, as-of statistical predictors and temporal splits.

Generic functions: the binary state may represent any market regime.
Docstrings are in English and no BTC-specific files are referenced here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def future_onset_labels(dates: pd.Series, states: pd.Series, horizon: int) -> pd.DataFrame:
    """Predict a new 0-to-1 state transition over a complete future horizon.

    Only currently negative, known states are eligible. An incomplete or
    missing future horizon is NA rather than a negative example. The outcome
    becomes available on the last day of that horizon.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    date = pd.to_datetime(dates, errors="raise").reset_index(drop=True)
    state = pd.Series(states).reset_index(drop=True)
    if len(date) != len(state) or date.isna().any() or date.duplicated().any():
        raise ValueError("Dates must be unique, nonmissing and match state length")
    if not date.is_monotonic_increasing:
        raise ValueError("Dates must be ordered")
    if not state.dropna().isin([0, 1, False, True]).all():
        raise ValueError("State must be binary or missing")
    arr = state.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(arr)
    starts = np.zeros(len(arr), dtype=bool)
    starts[1:] = valid[1:] & valid[:-1] & (arr[1:] == 1) & (arr[:-1] == 0)
    y = pd.Series(pd.NA, index=state.index, dtype="Int64")
    finish = pd.Series(pd.NaT, index=state.index, dtype="datetime64[ns]")
    eligible = valid & (arr == 0)
    for i in np.flatnonzero(eligible):
        right = i + horizon
        if right >= len(arr) or not valid[i + 1:right + 1].all():
            continue
        y.iloc[i] = int(starts[i + 1:right + 1].any())
        finish.iloc[i] = date.iloc[right]
    return pd.DataFrame({"date": date, "eligible": eligible, "onset": y, "label_end": finish})


def bsadf_asof_features(ref: pd.DataFrame) -> pd.DataFrame:
    """Compute last-observed BSADF excess, age and change with no future dates."""
    required = {"date", "reference_available_at", "bsadf_score_last",
                "bsadf_critical_last", "gsadf_evaluated_today"}
    if not required.issubset(ref):
        raise ValueError(f"Missing reference columns: {sorted(required - set(ref))}")
    d = ref[list(required)].copy().sort_values("date")
    d["date"] = pd.to_datetime(d.date)
    d["reference_available_at"] = pd.to_datetime(d.reference_available_at)
    ok = d.reference_available_at.notna()
    if (d.loc[ok, "reference_available_at"] > d.loc[ok, "date"]).any():
        raise ValueError("Reference computed using future market dates")
    d["bsadf_gap"] = d.bsadf_score_last - d.bsadf_critical_last
    d["bsadf_age_days"] = (d.date - d.reference_available_at).dt.days
    evaluated = d.bsadf_gap.where(d.gsadf_evaluated_today.eq(1))
    d["bsadf_gap_delta"] = (
        evaluated - evaluated.ffill().shift(1)
    ).where(d.gsadf_evaluated_today.eq(1)).ffill().fillna(0)
    return d[["date", "bsadf_gap", "bsadf_age_days", "bsadf_gap_delta"]]


def purged_year_split(data: pd.DataFrame, test_year: int, *,
                      date_col: str = "date", label_end_col: str = "label_end"):
    """Split train/validation/test, purging labels crossing either boundary."""
    d = data.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d[label_end_col] = pd.to_datetime(d[label_end_col])
    val_start = pd.Timestamp(f"{test_year - 1}-01-01")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    tr = d[(d[date_col] < val_start) & (d[label_end_col] < val_start)].copy()
    va = d[(d[date_col] >= val_start) & (d[date_col] < test_start) &
           (d[label_end_col] < test_start)].copy()
    te = d[(d[date_col] >= test_start) &
           (d[date_col] < pd.Timestamp(f"{test_year + 1}-01-01"))].copy()
    assert tr.empty or tr[label_end_col].max() < val_start
    assert va.empty or va[label_end_col].max() < test_start
    return tr, va, te


def select_alarm_threshold(y, scores, *, max_time_in_alarm: float = .30,
                           thresholds=None):
    """Choose balanced-accuracy optimum under a validation-only alarm budget.

    If no candidate is feasible, emit an always-off threshold and explicit
    failure status rather than silently choosing a test-dependent fallback.
    """
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(s) or not np.isfinite(s).all():
        return float("inf"), "invalid_validation"
    if np.unique(y).size < 2:
        return float("inf"), "one_class_validation"
    grid = list(thresholds) if thresholds is not None else np.arange(.05, .951, .05)
    options = []
    for t in grid:
        a = s >= t
        if a.mean() <= max_time_in_alarm:
            options.append((balanced_accuracy_score(y, a.astype(int)), -a.mean(), t))
    if not options:
        return float("inf"), "no_feasible_threshold"
    return float(sorted(options, reverse=True)[0][2]), "validated"
