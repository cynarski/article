"""Shared, auditable utilities for LPPLS + reference bubble-state experiments.

CAVEAT: The sparse-endpoint BSADF below is a sampled BSADF implementation with
an intercept-only, lag-0 right-tailed ADF regression and Monte Carlo random-walk
critical values. It is NOT interchangeable with a full PSY implementation using
other lag-selection/deterministic specifications.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, average_precision_score,
                             roc_auc_score, precision_recall_fscore_support,
                             balanced_accuracy_score)

UNION_EPISODES = [
    ("UNION01", "2016-06-08", "2016-06-18"),
    ("UNION02", "2016-12-30", "2017-01-15"),
    ("UNION03", "2017-02-22", "2017-03-07"),
    ("UNION04", "2017-04-27", "2017-07-05"),
    ("UNION05", "2017-07-15", "2018-01-09"),
    ("UNION06", "2019-06-17", "2019-07-13"),
    ("UNION07", "2019-08-09", "2019-08-10"),
    ("UNION08", "2019-12-06", "2020-01-05"),
    ("UNION09", "2020-02-07", "2020-02-17"),
    ("UNION10", "2020-03-20", "2020-09-21"),
    ("UNION11", "2020-11-06", "2021-05-20"),
    ("UNION12", "2021-08-08", "2021-09-19"),
    ("UNION13", "2021-10-05", "2021-12-02"),
    ("UNION14", "2024-03-04", "2024-03-25"),
]


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read genuine supplied closing-price CSV. Never generate synthetic fallback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Brak danych cenowych: {path}. Podaj prawdziwy plik BTC lub notowania indeksu.")
    d = pd.read_csv(path)
    dates = next((c for c in ('date', 'Date', 'Datetime', 'timestamp') if c in d), None)
    vals = next((c for c in ('price', 'Close', 'close', 'Adj Close') if c in d), None)
    if not dates or not vals:
        raise ValueError(f"Dane cenowe wymagają date/Date oraz price/Close; znaleziono {list(d.columns)}")
    z = d[[dates, vals]].rename(columns={dates:'date', vals:'price'}).copy()
    z['date'] = pd.to_datetime(z['date'], utc=True, errors='raise').dt.tz_convert(None).dt.normalize()
    z['price'] = pd.to_numeric(z['price'], errors='raise')
    z = z.sort_values('date').reset_index(drop=True)
    if z.date.duplicated().any() or not z.price.notna().all() or not z.price.gt(0).all():
        raise ValueError('Dane cenowe zawierają duplikaty, braki lub ceny niedodatnie.')
    if len(z) < 200: raise ValueError('Seria cen za krótka (<200 obserwacji).')
    return z


def union_labels(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    r = pd.Series(0, index=dates.index, dtype=int)
    for _, first, last in UNION_EPISODES:
        r |= dates.between(first, last).astype(int)
    return r


def _prefixes(x: np.ndarray):
    """Sufficient statistics for lag-0 ADF: delta y = intercept + beta*y[-1] + eps."""
    reg_x = x[:-1]
    reg_y = np.diff(x)
    sums = [reg_x, reg_y, reg_x*reg_x, reg_x*reg_y, reg_y*reg_y]
    return [np.r_[0., np.cumsum(a)] for a in sums]


def _adf_t_for_windows(pre, starts: np.ndarray, end: int) -> np.ndarray:
    """OLS right-tailed t-statistic for beta on windows [start, end] of log prices."""
    starts = np.asarray(starts, dtype=int)
    n = (end - starts).astype(float)
    sx, sz, sx2, sxz, sz2 = [v[end] - v[starts] for v in pre]
    sxx = sx2 - sx*sx/n
    sxz_center = sxz - sx*sz/n
    beta = np.divide(sxz_center, sxx, out=np.full_like(sxx, np.nan), where=sxx>1e-12)
    alpha = (sz-beta*sx)/n
    sse = sz2 - alpha*sz - beta*sxz
    sig2 = np.divide(sse, n-2, out=np.full_like(sse, np.nan), where=n>2)
    var_b = np.divide(sig2, sxx, out=np.full_like(sxx, np.nan), where=sxx>1e-12)
    se = np.sqrt(np.maximum(var_b, 0.))
    return np.divide(beta, se, out=np.full_like(beta, np.nan), where=se>1e-12)


def bsadf_sparsely_sampled(log_prices: np.ndarray, min_window=120,
                            endpoint_step=7, start_step=14, max_window=None,
                            evaluation_ends=None) -> pd.DataFrame:
    """Sampled BSADF: sup right-tailed ADF over candidate starts for each end.

    min_window counts PRICE observations, no ADF lagged differences beyond lag 0.
    If max_window is not None, this is a truncated rolling variant, not full BSADF.
    Evaluated only at designated end dates; last-known state may be held between ends.
    """
    x = np.asarray(log_prices, dtype=float)
    if len(x)<min_window: raise ValueError('Za krótki szereg dla wybranego min_window')
    if not np.isfinite(x).all(): raise ValueError('log_prices musi być skończony')
    ends = (np.arange(min_window-1, len(x), endpoint_step, dtype=int)
            if evaluation_ends is None else np.asarray(evaluation_ends, dtype=int))
    if ends.size and ends[-1]!=len(x)-1 and evaluation_ends is None:
        ends = np.r_[ends, len(x)-1]
    pre = _prefixes(x)
    out=[]
    for end in ends:
        lo = max(0, end-max_window+1) if max_window is not None else 0
        hi = end-min_window+1
        if hi<lo: continue
        starts=np.arange(lo, hi+1, start_step, dtype=int)
        if len(starts)==0 or starts[-1]!=hi: starts=np.r_[starts,hi]
        tstats=_adf_t_for_windows(pre, starts, int(end))
        best = float(np.nanmax(tstats)) if np.isfinite(tstats).any() else np.nan
        out.append((int(end), best, int(len(starts))))
    return pd.DataFrame(out, columns=['end_index','bsadf','n_windows_scanned'])


def monte_carlo_bsadf_reference(prices: pd.DataFrame, min_window=120, endpoint_step=7,
                                start_step=14, max_window=None, reps=99,
                                quantile=0.95, seed=42) -> tuple[pd.DataFrame,dict]:
    """Monte Carlo random-walk null; per-end empirical critical values; causal signal.

    Null innovations are iid N(0,1). This does not handle serial dependence,
    conditional heteroskedasticity, deterministic trends or data-driven selection.
    """
    if reps<19: raise ValueError('Użyj >=19 symulacji Monte Carlo')
    if not 0<quantile<1: raise ValueError('quantile musi należeć do (0,1)')
    x=np.log(prices.price.to_numpy(float))
    observed = bsadf_sparsely_sampled(x, min_window, endpoint_step, start_step, max_window)
    ends=observed.end_index.to_numpy(int)
    rng=np.random.default_rng(seed)
    simulated=np.empty((reps,len(ends)),dtype=float)
    for b in range(reps):
        rw=np.r_[0.0,np.cumsum(rng.standard_normal(len(x)-1))]
        series=bsadf_sparsely_sampled(rw,min_window,endpoint_step,start_step,max_window,evaluation_ends=ends)
        simulated[b,:]=series.bsadf.to_numpy(float)
    observed['critical_value']=np.nanquantile(simulated, quantile, axis=0)
    observed['evaluated_date']=prices.date.iloc[ends].to_numpy()
    observed['bubble_state_eval']=(observed.bsadf>observed.critical_value).astype(int)
    daily=prices[['date','price']].merge(observed[['evaluated_date','bsadf','critical_value','bubble_state_eval','n_windows_scanned']],
                                left_on='date',right_on='evaluated_date',how='left',validate='one_to_one')
    daily['gsadf_evaluated_today']=daily.evaluated_date.notna().astype(int)
    daily['bsadf_score_last']=daily.bsadf.ffill()
    daily['bsadf_critical_last']=daily.critical_value.ffill()
    daily['bubble_state_gsadf']=daily.bubble_state_eval.ffill().astype('Int64')
    daily['reference_available_at']=daily.evaluated_date.ffill()
    daily['bubble_union']=union_labels(daily.date)
    daily=daily.drop(columns=['evaluated_date'])
    settings=dict(method='Sampled expanding BSADF (ADF(0), intercept); Monte Carlo random-walk null',
                  min_window=int(min_window),endpoint_step=int(endpoint_step),start_step=int(start_step),
                  max_window=max_window,mc_reps=int(reps),critical_quantile=quantile,seed=int(seed),
                  n_price_rows=len(prices),n_evaluated_endpoints=len(observed))
    return daily, settings


def build_daily_features(prices: pd.DataFrame, fits_file: str|Path) -> tuple[pd.DataFrame,dict]:
    """LPPLS features as of each day; no retrospective UNION or future targets as inputs."""
    d=prices[['date','price']].copy().sort_values('date').reset_index(drop=True)
    logp=np.log(d.price)
    d['ret_1d']=logp.diff()
    for k in [7,14,30,60]: d[f'momentum_{k}']=logp.diff(k)
    for k in [7,30,60]: d[f'vol_{k}']=d.ret_1d.rolling(k,min_periods=k).std()
    for k in [30,90]: d[f'drawdown_{k}']=d.price/d.price.rolling(k,min_periods=k).max()-1
    fit=pd.read_csv(fits_file)
    required={'win_end','tc','m','w','B','C','r2','n','ok'}
    if not required.issubset(fit): raise ValueError(f'Brak kolumn w dopasowaniach: {required-set(fit.columns)}')
    fit['win_end']=pd.to_datetime(fit.win_end,utc=True).dt.tz_convert(None).dt.normalize()
    for c in ['tc','m','w','B','C','r2','n']:
        fit[c]=pd.to_numeric(fit[c],errors='coerce')
    fit['t_gap']=fit.tc-fit.win_end.map(lambda x:x.toordinal())
    ok=fit.ok.astype(str).str.lower().isin(['true','1','yes'])
    fit['accepted']=(ok & fit.m.gt(0)&fit.m.lt(1)&fit.w.between(6,13)&fit.B.lt(0)&
                     fit.t_gap.gt(0)&fit.t_gap.le(40)&fit.r2.gt(0.90))
    n_all=fit.groupby('win_end').size().rename('lppls_n_all')
    n_ok=fit.groupby('win_end').accepted.sum().rename('lppls_n_accepted')
    agg=fit.loc[fit.accepted].groupby('win_end').agg(
        lppls_m=('m','median'),lppls_w=('w','median'),lppls_tgap=('t_gap','median'),
        lppls_r2=('r2','median'),lppls_B=('B','median'),lppls_C=('C','median'),
        lppls_window_n=('n','median'),lppls_tgap_std=('t_gap','std'))
    d=d.merge(n_all,left_on='date',right_index=True,how='left',validate='one_to_one')
    d=d.merge(n_ok,left_on='date',right_index=True,how='left',validate='one_to_one')
    d=d.merge(agg,left_on='date',right_index=True,how='left',validate='one_to_one')
    d['lppls_n_all']=d.lppls_n_all.fillna(0).astype(int)
    d['lppls_n_accepted']=d.lppls_n_accepted.fillna(0).astype(int)
    d['lppls_ci']=np.divide(d.lppls_n_accepted,d.lppls_n_all,
                            out=np.zeros(len(d),dtype=float),where=d.lppls_n_all>0)
    price_cols=['ret_1d']+[f'momentum_{k}' for k in [7,14,30,60]]+[f'vol_{k}' for k in [7,30,60]]+[f'drawdown_{k}' for k in [30,90]]
    lppls_cols=['lppls_ci','lppls_n_accepted']+list(agg.columns)
    feat={'Price-only':price_cols,'LPPLS-only':lppls_cols,'Price+LPPLS':price_cols+lppls_cols}
    return d,feat


def classification_stats(y,pred,score=None):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    precision=tp/(tp+fp) if tp+fp else np.nan
    recall=tp/(tp+fn) if tp+fn else np.nan
    specificity=tn/(tn+fp) if tn+fp else np.nan
    f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan
    res={'n':len(y),'positives':int(y.sum()),'positive_rate':float(y.mean()),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),
         'precision':precision,'recall':recall,'specificity':specificity,'f1':f1,
         'balanced_accuracy':float((recall+specificity)/2) if np.isfinite(recall) and np.isfinite(specificity) else np.nan,
         'time_in_alarm':float(pred.mean())}
    if score is not None:
        res['AP']=float(average_precision_score(y,score)) if y.sum()>0 else np.nan
        res['ROC_AUC']=float(roc_auc_score(y,score)) if len(np.unique(y))==2 else np.nan
    return res


def future_drop_labels(prices: pd.DataFrame, thresholds=(10,15,20), horizons=(30,45,60)):
    """Labels relative to P_t using next H observed bars; no partial right-edge horizons."""
    d=prices[['date','price']].copy()
    for h in horizons:
        future_min=pd.concat([d.price.shift(-k) for k in range(1,h+1)],axis=1).min(axis=1)
        end=d.date.shift(-h)
        full=end.notna()
        d[f'future_min_drop_{h}']=(future_min/d.price-1).where(full)
        d[f'label_end_{h}']=end
        for thr in thresholds:
            d[f'drop_{thr}_h{h}']=(d[f'future_min_drop_{h}']<=-thr/100).where(full).astype('Int64')
    return d


def onsets(d: pd.DataFrame, alarm_col='alarm') -> pd.DataFrame:
    """Only actual transitions 0->1; leftmost alarm excluded from onset count."""
    z=d[['date',alarm_col]].sort_values('date').copy().reset_index(drop=True)
    z['previous']=z[alarm_col].shift(1)
    z['left_censored']=(z.index==0)&z[alarm_col].eq(1)
    z['onset']=(z[alarm_col].eq(1)&z.previous.eq(0)).astype(int)
    return z


def fit_stock_lppls(prices:pd.DataFrame, lookbacks=(90,120,180),step=10,
                    n_starts=3,seed=42,max_windows=None):
    """Optional stock-only calibration (calendar-day tc, observed bar lookbacks).

    Uses linear/nonlinear separable LPPLS and multi-start L-BFGS-B. A reference
    implementation for cross-market robustness, NOT identical to prior BTC fitter.
    """
    from scipy.optimize import minimize
    rng=np.random.default_rng(seed)
    d=prices.sort_values('date').reset_index(drop=True)
    ln=np.log(d.price.to_numpy())
    ords=d.date.map(lambda x:x.toordinal()).to_numpy(float)
    rows=[]
    for n in lookbacks:
        for end in range(n-1,len(d),step):
            if max_windows is not None and len(rows)>=max_windows: break
            i=end-n+1
            t=ords[i:end+1]; y=ln[i:end+1]; te=t[-1]
            def solve(z):
                gap,m,w=z; tau=te+gap-t
                if np.any(tau<=0): return np.inf,None,None
                f=tau**m
                M=np.column_stack([np.ones(len(t)),f,f*np.cos(w*np.log(tau)),f*np.sin(w*np.log(tau))])
                try: coef=np.linalg.lstsq(M,y,rcond=None)[0]
                except np.linalg.LinAlgError: return np.inf,None,None
                residual=y-M@coef
                return float(np.dot(residual,residual)),coef,M
            best=(np.inf,None,None)
            for _ in range(n_starts):
                x0=[rng.uniform(2,38),rng.uniform(.15,.85),rng.uniform(6,13)]
                try:
                    fit=minimize(lambda z:solve(z)[0],x0,method='L-BFGS-B',
                                 bounds=[(1,40),(.05,.95),(5,15)],options={'maxiter':100})
                    ssr,coef,_=solve(fit.x)
                    if ssr<best[0] and np.isfinite(ssr):best=(ssr,fit.x.copy(),coef.copy())
                except Exception:
                    continue
            ssr,z,coef=best
            ok=z is not None
            denom=float(np.sum((y-y.mean())**2))
            rows.append({'win_start':d.date.iloc[i],'win_end':d.date.iloc[end],
                         'n':int(n),'ok':bool(ok),'tc':(te+z[0]) if ok else np.nan,
                         'm':z[1] if ok else np.nan,'w':z[2] if ok else np.nan,
                         'B':coef[1] if ok else np.nan,
                         'C':float(np.hypot(coef[2],coef[3])) if ok else np.nan,
                         'r2':(1-ssr/denom) if ok and denom>0 else np.nan,'ssr':ssr if ok else np.nan})
        if max_windows is not None and len(rows)>=max_windows:break
    return pd.DataFrame(rows)
