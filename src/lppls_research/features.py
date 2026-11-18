"""Reusable daily aggregation for rolling model fits, including LPPLS adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def _as_dates(values: pd.Series, name: str) -> pd.Series:
    """Convert date-like values into timezone-naive normalized timestamps."""
    result = pd.to_datetime(values, utc=True, errors="raise").dt.tz_convert(None).dt.normalize()
    if result.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    return result


@dataclass(frozen=True)
class LPPLSAcceptanceCriteria:
    """Configure a positive-bubble LPPLS filter without coupling it to a dataset.

    The critical time is expected as a numeric proleptic Gregorian ordinal,
    matching pandas.Timestamp.toordinal(). This class does not fit LPPLS.
    """

    m_lower: float = 0.0
    m_upper: float = 1.0
    omega_lower: float = 6.0
    omega_upper: float = 13.0
    critical_gap_min: float = 0.0
    critical_gap_max: float = 40.0
    r2_min: float = 0.90

    def evaluate(self, fits: pd.DataFrame, *, end_col: str = "win_end") -> pd.DataFrame:
        """Return fit records with calculated t_gap and accepted columns.

        Input rows are preserved, including failed and rejected fits. All
        comparisons use only fit parameters and the corresponding window end.
        """
        required = {end_col, "ok", "tc", "m", "w", "B", "r2"}
        missing = required.difference(fits.columns)
        if missing:
            raise ValueError(f"Missing fit columns: {sorted(missing)}")
        data = fits.copy()
        dates = _as_dates(data[end_col], end_col)
        for name in ("tc", "m", "w", "B", "r2"):
            data[name] = pd.to_numeric(data[name], errors="coerce")
        if pd.api.types.is_bool_dtype(data["ok"]):
            succeeded = data["ok"].fillna(False)
        else:
            succeeded = data["ok"].astype("string").str.strip().str.lower().isin(("true", "1", "yes"))
        ordinals = dates.map(lambda value: value.toordinal()).to_numpy(dtype=float)
        data["t_gap"] = data["tc"] - ordinals
        data["accepted"] = (
            succeeded & data["m"].gt(self.m_lower) & data["m"].lt(self.m_upper)
            & data["w"].between(self.omega_lower, self.omega_upper)
            & data["B"].lt(0)
            & data["t_gap"].gt(self.critical_gap_min)
            & data["t_gap"].le(self.critical_gap_max)
            & data["r2"].gt(self.r2_min)
        ).fillna(False).astype(bool)
        data[end_col] = dates
        return data


class RollingFitAggregator:
    """Aggregate any overlapping-window fits into one row per calendar date.

    The acceptance decision may come from a model-specific rule or a stored
    boolean column. Denominators count all fits, including failed fits.
    Metrics are summarized exclusively over accepted fits. Missing-fit days
    remain in the calendar with zero counts and missing metric summaries.

    Parameters
    ----------
    end_col:
        Date when each historical fit becomes available.
    accepted_col:
        Boolean acceptance indicator already present in the input data.
    metrics:
        Mapping from output feature name to (source_column, aggregation).
    prefix:
        Prefix for count and accepted-fraction output columns.
    """

    def __init__(self, *, end_col: str = "win_end", accepted_col: str = "accepted",
                 metrics: Mapping[str, tuple[str, str]] | None = None,
                 prefix: str = "fit") -> None:
        if not prefix or not prefix.isidentifier():
            raise ValueError("prefix must be a non-empty Python identifier")
        self.end_col = end_col
        self.accepted_col = accepted_col
        self.metrics = dict(metrics or {})
        self.prefix = prefix

    def transform(self, fits: pd.DataFrame, *, calendar: pd.Series | None = None) -> pd.DataFrame:
        """Create a dated feature frame using an optional full prediction calendar."""
        required = {self.end_col, self.accepted_col}
        required.update(source for source, _ in self.metrics.values())
        missing = required.difference(fits.columns)
        if missing:
            raise ValueError(f"Missing aggregation columns: {sorted(missing)}")
        frame = fits.copy()
        frame[self.end_col] = _as_dates(frame[self.end_col], self.end_col)
        accepted = frame[self.accepted_col]
        if not pd.api.types.is_bool_dtype(accepted):
            normalized = accepted.astype("string").str.strip().str.lower()
            if (~normalized.isin(["true", "false", "1", "0"])).any():
                raise ValueError("Accepted-fit indicator must contain only booleans")
            accepted = normalized.isin(["true", "1"])
        if accepted.isna().any():
            raise ValueError("Accepted-fit indicator must not contain missing values")
        frame[self.accepted_col] = accepted.astype(bool)
        if calendar is None:
            index = pd.DatetimeIndex(sorted(frame[self.end_col].unique()), name="date")
        else:
            cal = _as_dates(pd.Series(calendar), "calendar")
            if cal.duplicated().any():
                raise ValueError("Prediction calendar contains duplicate dates")
            index = pd.DatetimeIndex(cal.sort_values(), name="date")
            if (~frame[self.end_col].isin(index)).any():
                raise ValueError("Fit end dates fall outside the supplied calendar")
        counts = frame.groupby(self.end_col).size().reindex(index, fill_value=0)
        accepted_counts = frame.groupby(self.end_col)[self.accepted_col].sum().reindex(index, fill_value=0)
        n_all = f"{self.prefix}_n_all"
        n_acc = f"{self.prefix}_n_accepted"
        ci = f"{self.prefix}_ci"
        result = pd.DataFrame(index=index)
        result[n_all] = counts.astype(int)
        result[n_acc] = accepted_counts.astype(int)
        result[ci] = np.divide(result[n_acc], result[n_all],
                               out=np.zeros(len(result), dtype=float), where=result[n_all] > 0)
        if self.metrics:
            for source, _ in self.metrics.values():
                frame[source] = pd.to_numeric(frame[source], errors="coerce")
            aggregated = frame.loc[frame[self.accepted_col]].groupby(self.end_col).agg(**{
                name: pd.NamedAgg(column=source, aggfunc=func)
                for name, (source, func) in self.metrics.items()
            })
            result = result.join(aggregated)
        return result.reset_index()


LPPLS_METRICS: dict[str, tuple[str, str]] = {
    "lppls_m": ("m", "median"), "lppls_w": ("w", "median"),
    "lppls_tgap": ("t_gap", "median"), "lppls_r2": ("r2", "median"),
    "lppls_B": ("B", "median"), "lppls_C": ("C", "median"),
    "lppls_window_n": ("n", "median"),
    "lppls_tgap_std": ("t_gap", "std"),
}


def build_lppls_daily_features(fits: pd.DataFrame, *, calendar: pd.Series | None = None,
                               criteria: LPPLSAcceptanceCriteria | None = None) -> pd.DataFrame:
    """Reproduce the project's daily LPPLS columns from complete raw fit records.

    Do not prefilter the input: failed and rejected fits contribute to the
    Confidence Indicator denominator. The caller supplies the decision-date
    calendar to preserve days without accepted or attempted fits.
    """
    accepted_fits = (criteria or LPPLSAcceptanceCriteria()).evaluate(fits)
    return RollingFitAggregator(metrics=LPPLS_METRICS, prefix="lppls").transform(
        accepted_fits, calendar=calendar
    )
