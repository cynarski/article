"""Publication timelines for seven chronological, out-of-sample warning methods.

LPPLS-CI and BSADF-gap are statistical rules from a fixed-configuration
benchmark. Five machine-learning families use Price+LPPLS features from
chronologically tuned models. These protocols are distinct: plots compare
warning periods, *not* equal-budget predictive performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METHODS = [
    ("LPPLS Confidence Indicator", "fixed", "StatisticalRule", "LPPLS-CI"),
    ("BSADF statistical rule", "fixed", "StatisticalRule", "BSADF-gap"),
    ("Logistic Regression", "tuned", "LogisticRegression", "Price+LPPLS"),
    ("Decision Tree", "tuned", "DecisionTree", "Price+LPPLS"),
    ("XGBoost", "tuned", "XGBoost", "Price+LPPLS"),
    ("LightGBM", "tuned", "LightGBM", "Price+LPPLS"),
    ("CatBoost", "tuned", "CatBoost", "Price+LPPLS"),
]


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort dates and fail instead of silently deduplicating forecasts."""
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result = result.sort_values("date").reset_index(drop=True)
    if result.date.duplicated().any():
        raise ValueError("Duplicate OOS dates")
    return result


def aligned_alarms(tuned: pd.DataFrame, fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align exact OOS dates and identical future-drop labels across methods."""
    needed_tuned = {"date", "drop_pct", "horizon", "model", "feature_group", "y_true", "y_pred"}
    needed_fixed = {"date", "model", "feature_group", "target", "alarm"}
    if not needed_tuned.issubset(tuned) or not needed_fixed.issubset(fixed):
        raise ValueError("Missing required prediction fields")
    tuned = tuned[tuned.drop_pct.eq(10) & tuned.horizon.eq(60)]
    dates, targets, alarms, metadata = None, None, None, []
    for index, (label, source, model, group) in enumerate(METHODS):
        origin = fixed if source == "fixed" else tuned
        subset = origin[origin.model.eq(model) & origin.feature_group.eq(group)]
        if subset.empty:
            raise ValueError(f"Missing OOS predictions for {label}")
        subset = _normalize(subset)
        truth = subset["target" if source == "fixed" else "y_true"].to_numpy()
        signal = subset["alarm" if source == "fixed" else "y_pred"].to_numpy()
        if not np.isin(truth, [0, 1]).all() or not np.isin(signal, [0, 1]).all():
            raise ValueError(f"Nonbinary predictions for {label}")
        if not subset.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Nonconsecutive OOS dates for {label}")
        if dates is None:
            dates, targets = subset.date, truth
            alarms = pd.DataFrame({"date": dates, "target": targets})
        elif not dates.equals(subset.date) or not np.array_equal(targets, truth):
            raise ValueError(f"Inconsistent OOS date/target for {label}")
        alarms[f"alarm_{index}"] = signal.astype(int)
        metadata.append({"method": label, "source": source, "feature_group": group,
                         "n_days": len(subset), "n_positive": int(truth.sum()),
                         "time_in_alarm_pct": float(signal.mean() * 100),
                         "n_alarm_days": int(signal.sum())})
    if len(alarms) != 1401:
        raise ValueError(f"Expected 1401 precomputed BTC OOS dates, got {len(alarms)}")
    return alarms, pd.DataFrame(metadata)


def contiguous_intervals(date: pd.Series, markers: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert consecutive positive daily states to half-open date intervals."""
    markers = np.asarray(markers, dtype=int)
    starts = np.flatnonzero((markers == 1) & ~np.r_[False, markers[:-1] == 1])
    ends = np.flatnonzero((markers == 1) & ~np.r_[markers[1:] == 1, False])
    d = pd.Series(pd.to_datetime(date)).reset_index(drop=True)
    return [(d.iloc[a], d.iloc[b] + pd.Timedelta(days=1)) for a, b in zip(starts, ends)]


def render(alarms: pd.DataFrame, reference: pd.DataFrame, out: Path) -> None:
    """Draw a compact chart and seven source-style repeated price panels."""
    ref = _normalize(reference)
    required = {"price", "bubble_union"}
    if not required.issubset(ref):
        raise ValueError("Reference lacks BTC price or UNION periods")
    timeline = alarms[["date"]].merge(ref[["date", "price", "bubble_union"]],
                                      on="date", how="left", validate="one_to_one")
    if timeline[["price", "bubble_union"]].isna().any().any():
        raise ValueError("Missing price or retrospective UNION for OOS dates")
    union = contiguous_intervals(timeline.date, timeline.bubble_union.to_numpy())
    out.mkdir(parents=True, exist_ok=True)
    legend = [Line2D([0], [0], color="black", label="BTC-USD close"),
              Patch(color="#ed9c9c", alpha=.30, label="Retrospective UNION"),
              Patch(color="#7b8795", alpha=.48, label="Stored OOS alarm")]

    fig, (ax, alarms_ax) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2.1, 2]}, layout="constrained")
    for a, b in union:
        ax.axvspan(a, b, color="#ed9c9c", alpha=.28, lw=0)
        alarms_ax.axvspan(a, b, color="#ed9c9c", alpha=.14, lw=0)
    ax.plot(timeline.date, timeline.price, color="black", lw=1)
    ax.set(ylabel="BTC-USD price (USD)",
           title="Out-of-sample early-warning periods across seven methods")
    ax.legend(handles=legend, loc="upper left", ncol=3, fontsize=8)
    for i, (label, _, _, _) in enumerate(METHODS):
        for a, b in contiguous_intervals(alarms.date, alarms[f"alarm_{i}"].to_numpy()):
            alarms_ax.broken_barh([(mdates.date2num(a), (b-a).days)],
                                  (i-.35, .7), facecolors="#657688")
    alarms_ax.set_yticks(range(len(METHODS)), [m[0] for m in METHODS], fontsize=8.3)
    alarms_ax.set_ylim(len(METHODS)-.5, -.6)
    alarms_ax.set(xlabel="Date", ylabel="OOS warning state")
    alarms_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    alarms_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(alpha=.2)
    alarms_ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "oos_seven_methods_compact.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, panels = plt.subplots(7, 1, figsize=(12, 12.5), sharex=True, layout="constrained")
    for i, (panel, (label, _, _, _)) in enumerate(zip(panels, METHODS)):
        for a, b in union:
            panel.axvspan(a, b, color="#ed9c9c", alpha=.25, lw=0)
        for a, b in contiguous_intervals(alarms.date, alarms[f"alarm_{i}"].to_numpy()):
            panel.axvspan(a, b, color="#7b8795", alpha=.36, lw=0)
        panel.plot(timeline.date, timeline.price, color="black", lw=.8)
        panel.set_title(label, loc="left", fontsize=9)
        panel.set_ylabel("USD", fontsize=8)
        panel.tick_params(labelsize=8)
        panel.grid(alpha=.17)
    panels[0].legend(handles=legend, loc="upper left", ncol=3, fontsize=8)
    panels[-1].set_xlabel("Date")
    panels[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    panels[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("BTC-USD and precomputed OOS warnings: 10% decline in 60 days")
    fig.savefig(out / "oos_seven_methods_price_panels.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate figures and an audit table without re-fitting any estimator."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-08b", required=True, type=Path)
    p.add_argument("--pred-18", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    forecasts, metadata = aligned_alarms(pd.read_csv(args.pred_08b),
                                         pd.read_csv(args.pred_18))
    render(forecasts, pd.read_csv(args.reference), args.out)
    forecasts.to_csv(args.out / "oos_seven_method_daily_alarms.csv", index=False)
    metadata.to_csv(args.out / "oos_seven_method_metadata.csv", index=False)
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()
