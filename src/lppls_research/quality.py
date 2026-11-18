"""General data-lineage checks for rolling fits and temporal ML experiments.

No dependency on BTC, an online price service, or an ML estimator is required.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd

from .availability import TemporalAvailabilityValidator
from .features import LPPLSAcceptanceCriteria, build_lppls_daily_features


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file to calculate a reproducible SHA-256 identifier."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prices(frame: pd.DataFrame, *, date_col: str | None = None,
                     price_col: str | None = None,
                     require_daily: bool = False) -> pd.DataFrame:
    """Return validated date/price columns without interpolating observations.

    The optional daily-calendar requirement is appropriate for crypto assets.
    It is deliberately disabled for stock-index trading calendars.
    """
    dc = date_col or next((c for c in ("date","Date","Datetime","timestamp") if c in frame),None)
    pc = price_col or next((c for c in ("price","Close","close","Adj Close") if c in frame),None)
    if not dc or not pc or dc == pc:
        raise ValueError(f"Cannot identify date/price fields: {list(frame)}")
    out = frame[[dc,pc]].rename(columns={dc:"date",pc:"price"}).copy()
    out["date"] = pd.to_datetime(out.date,utc=True,errors="raise").dt.tz_convert(None).dt.normalize()
    out["price"] = pd.to_numeric(out.price,errors="raise")
    out = out.sort_values("date").reset_index(drop=True)
    TemporalAvailabilityValidator.check_calendar(out,require_consecutive_days=require_daily)
    if not np.isfinite(out.price).all() or not out.price.gt(0).all():
        raise ValueError("Prices must be finite and strictly positive")
    return out


def verify_fits(fits: pd.DataFrame, *, expected_rows: int | None = None,
                date_col: str = "win_end") -> dict:
    """Validate rolling-window fit records and compute acceptance diagnostics."""
    required={"win_start",date_col,"ok","tc","m","w","B","C","r2","n"}
    missing=required-set(fits)
    if missing: raise ValueError(f"Missing fit fields: {sorted(missing)}")
    TemporalAvailabilityValidator.check_windows(fits,start_col="win_start",end_col=date_col)
    if expected_rows is not None and len(fits)!=expected_rows:
        raise ValueError(f"Expected {expected_rows} fits, got {len(fits)}")
    if fits.duplicated(["win_start",date_col]).any():
        raise ValueError("Duplicate rolling windows")
    accepted=LPPLSAcceptanceCriteria().evaluate(fits,end_col=date_col)
    ends=pd.to_datetime(fits[date_col],utc=True)
    return {"all_fits":int(len(fits)),"accepted_fits":int(accepted.accepted.sum()),
            "acceptance_rate":float(accepted.accepted.mean()),
            "date_min":str(ends.min().date()),"date_max":str(ends.max().date()),
            "unique_end_dates":int(ends.nunique())}


def ensure_same_daily_features(left: pd.DataFrame, right: pd.DataFrame,
                               *, date_col: str = "date", columns=None,
                               atol: float = 1e-10) -> dict:
    """Require exactly matching dates and numerically equivalent features."""
    fields=list(columns or ["lppls_ci","lppls_n_accepted","lppls_m","lppls_w",
            "lppls_tgap","lppls_r2","lppls_B","lppls_C",
            "lppls_window_n","lppls_tgap_std"])
    a,b=left.copy(),right.copy()
    for frame in (a,b):
        frame[date_col]=pd.to_datetime(frame[date_col],utc=True).dt.tz_convert(None).dt.normalize()
        if frame[date_col].duplicated().any():
            raise ValueError("Duplicate feature dates")
    if set(a[date_col]) != set(b[date_col]):
        raise ValueError("The two feature datasets have different date sets")
    missing=(set(fields)|{date_col})-(set(a)&set(b))
    if missing:raise ValueError(f"Missing fields: {sorted(missing)}")
    joined=a[[date_col]+fields].merge(b[[date_col]+fields],on=date_col,
                 suffixes=("_a","_b"),validate="one_to_one")
    for field in fields:
        va=pd.to_numeric(joined[field+"_a"],errors="raise").to_numpy(float)
        vb=pd.to_numeric(joined[field+"_b"],errors="raise").to_numpy(float)
        equal=np.isclose(va,vb,atol=atol,rtol=0,equal_nan=True)
        if not equal.all():
            raise ValueError(f"{field} differs on {int((~equal).sum())} dates")
    return {"dates":int(len(joined)),"features_compared":len(fields),"mismatches":0}


def audit_input_files(price_file: str | Path, fits_file: str | Path,
                      *, expected_fits: int | None = None,
                      require_daily_prices: bool = False) -> dict:
    """Audit input hashes, all fits, and daily feature coverage without refitting."""
    pp,fp=Path(price_file),Path(fits_file)
    if not pp.is_file() or not fp.is_file():
        raise FileNotFoundError(f"Missing input file: {pp} or {fp}")
    prices=normalize_prices(pd.read_csv(pp),require_daily=require_daily_prices)
    fits=pd.read_csv(fp)
    stats=verify_fits(fits,expected_rows=expected_fits)
    features=build_lppls_daily_features(fits,calendar=prices.date)
    if int(features.lppls_n_all.sum())!=len(fits):
        raise ValueError("Some fit windows are missing from the prediction calendar")
    return {
        "prices":{"path":str(pp),"sha256":sha256_file(pp),
                  "rows":len(prices),"first_date":str(prices.date.min().date()),
                  "last_date":str(prices.date.max().date())},
        "fits":{"path":str(fp),"sha256":sha256_file(fp),**stats},
        "daily":{"dates":len(features),
                 "days_with_any_fits":int(features.lppls_n_all.gt(0).sum()),
                 "days_with_accepted_fits":int(features.lppls_n_accepted.gt(0).sum())},
        "filter":asdict(LPPLSAcceptanceCriteria()),
    }
