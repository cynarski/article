"""Temporal-data availability checks for causal machine-learning experiments."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class TemporalAvailabilityValidator:
    """Validate when features, labels, and chronological partitions are usable.

    The validator is model-agnostic and supports irregular trading calendars.
    It makes no assumption that a fixed embargo is required for every target:
    the actual label-availability timestamps determine whether a split leaks.
    """

    @staticmethod
    def _times(frame: pd.DataFrame, column: str, *, allow_missing: bool = False) -> pd.Series:
        """Parse timestamps while preserving missing values for optional references."""
        if column not in frame:
            raise ValueError(f"Missing timestamp column: {column}")
        times = pd.to_datetime(frame[column], utc=True, errors="raise").dt.tz_convert(None)
        if not allow_missing and times.isna().any():
            raise ValueError(f"Missing timestamps in {column}")
        return times

    @classmethod
    def check_calendar(cls, frame: pd.DataFrame, *, date_col: str = "date",
                       require_consecutive_days: bool = False) -> None:
        """Reject missing, duplicated, unordered, or optionally skipped dates."""
        dates = cls._times(frame, date_col)
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{date_col} must be unique and ordered")
        if require_consecutive_days and len(dates) > 1:
            if not dates.diff().dropna().eq(pd.Timedelta(days=1)).all():
                raise ValueError("Calendar contains nonconsecutive dates")

    @classmethod
    def check_as_of(cls, frame: pd.DataFrame, *, available_col: str,
                    decision_col: str = "date", allow_missing: bool = False,
                    strict: bool = False) -> None:
        """Require information to be available no later than its decision time."""
        decision = cls._times(frame, decision_col)
        available = cls._times(frame, available_col, allow_missing=allow_missing)
        valid = available.notna()
        leaked = available[valid].ge(decision[valid]) if strict else available[valid].gt(decision[valid])
        if leaked.any():
            first = frame.index[valid][leaked.to_numpy()][0]
            raise ValueError(f"Future information in {available_col} at row {first}")

    @classmethod
    def check_windows(cls, fits: pd.DataFrame, *, start_col: str = "win_start",
                      end_col: str = "win_end") -> None:
        """Require each rolling window to have a nonnegative historical extent."""
        start = cls._times(fits, start_col)
        end = cls._times(fits, end_col)
        if (start > end).any():
            raise ValueError("Found rolling fit with start later than end")

    @classmethod
    def check_split(cls, train: pd.DataFrame, validation: pd.DataFrame,
                    test: pd.DataFrame, *, date_col: str = "date",
                    label_end_col: str | None = None,
                    feature_ready_col: str | None = None) -> None:
        """Enforce strictly ordered folds and fully observable past labels.

        If label_end_col is provided, labels in a preceding partition must
        become known strictly before the next partition begins. For a current
        state label known on its date, pass its as-of column or omit this check.
        """
        if any(part.empty for part in (train, validation, test)):
            raise ValueError("Train, validation, and test must all be nonempty")
        for part in (train, validation, test):
            cls.check_calendar(part, date_col=date_col)
            if feature_ready_col is not None:
                cls.check_as_of(part, available_col=feature_ready_col, decision_col=date_col)
        tr_dates = cls._times(train, date_col)
        va_dates = cls._times(validation, date_col)
        te_dates = cls._times(test, date_col)
        if not (tr_dates.max() < va_dates.min() and va_dates.max() < te_dates.min()):
            raise ValueError("Chronological partitions overlap or are unordered")
        if label_end_col is not None:
            for earlier, next_start in ((train, va_dates.min()), (validation, te_dates.min())):
                label_end = cls._times(earlier, label_end_col)
                if label_end.ge(next_start).any():
                    raise ValueError(f"{label_end_col} leaks into a later partition")

    @classmethod
    def check_matching_counts(cls, reference: pd.DataFrame, actual: pd.DataFrame,
                              *, mapping: Mapping[str, str], date_col: str = "date",
                              atol: float = 1e-12) -> None:
        """Require daily count/fraction columns from two pipelines to agree.

        mapping maps reference-column names to actual-column names.
        All reference dates must occur in the computed daily dataset.
        """
        for frame in (reference, actual):
            if frame[date_col].duplicated().any():
                raise ValueError("Cannot compare frames with duplicated dates")
        left = reference.copy()
        right = actual.copy()
        left[date_col] = cls._times(left, date_col).dt.normalize()
        right[date_col] = cls._times(right, date_col).dt.normalize()
        fields = {date_col, *mapping.keys()}
        targets = {date_col, *mapping.values()}
        if not fields.issubset(left) or not targets.issubset(right):
            raise ValueError("One of the comparison columns is missing")
        left_cols = {old: f"_ref_{index}" for index, old in enumerate(mapping)}
        right_cols = {new: f"_new_{index}" for index, new in enumerate(mapping.values())}
        joined = left[list(fields)].rename(columns=left_cols).merge(
            right[list(targets)].rename(columns=right_cols), on=date_col,
            how="left", validate="one_to_one", indicator=True
        )
        if joined._merge.ne("both").any():
            raise ValueError("Some reference dates have no computed features")
        for index, (old, new) in enumerate(mapping.items()):
            a = pd.to_numeric(joined[f"_ref_{index}"], errors="raise").to_numpy(dtype=float)
            b = pd.to_numeric(joined[f"_new_{index}"], errors="raise").to_numpy(dtype=float)
            equal = np.isclose(a, b, atol=atol, rtol=0, equal_nan=True)
            if not equal.all():
                raise ValueError(f"{old} != {new} on {int((~equal).sum())} dates")
