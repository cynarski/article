"""Plot publication-ready OOS error matrices and event-warning trade-offs.

All model alarms must be loaded from precomputed chronological OOS forecasts;
this script does not select test-time thresholds or re-train estimators.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

MODELS = [
    ("XGBoost", "Price-only", "Price only"),
    ("XGBoost", "Price+LPPLS", "Price + LPPLS"),
    ("XGBoost", "Price+BSADF", "Price + BSADF"),
    ("XGBoost", "Price+LPPLS+BSADF", "Full hybrid"),
    ("StatisticalRule", "BSADF-gap", "BSADF rule"),
    ("StatisticalRule", "LPPLS-CI", "LPPLS CI rule"),
]

def matrix_counts(y, alarm):
    """Return TN, FP, FN, TP for two equally long binary vectors."""
    y, alarm = np.asarray(y, dtype=int), np.asarray(alarm, dtype=int)
    if len(y) != len(alarm) or not set(y) <= {0,1} or not set(alarm) <= {0,1}:
        raise ValueError("Expected equal-length binary vectors")
    return (int(((y==0)&(alarm==0)).sum()), int(((y==0)&(alarm==1)).sum()),
            int(((y==1)&(alarm==0)).sum()), int(((y==1)&(alarm==1)).sum()))

def check_predictions(d, target="target", alarm="alarm"):
    """Reject missing, duplicated, nonchronological or nonbinary forecast rows."""
    for field in ["date","model","feature_group",target,alarm]:
        if field not in d: raise ValueError(f"Missing field {field}")
    d = d.copy()
    d["date"] = pd.to_datetime(d.date)
    for key, part in d.groupby(["model","feature_group"]):
        part = part.sort_values("date")
        if part.date.duplicated().any() or not part.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"Non-continuous or duplicated OOS dates for {key}")
        matrix_counts(part[target],part[alarm])
    return d

def drawdown_events(prices, threshold=.10, lookback=180, separation=60):
    """Find first 10% threshold crossings against a strictly historical high."""
    d = prices.sort_values("date")
    peak = d.price.rolling(lookback,min_periods=lookback).max().shift(1)
    below = ((d.price / peak - 1) <= -threshold).fillna(False).to_numpy()
    crossing = np.flatnonzero(below & ~np.r_[False, below[:-1]])
    events=[]
    for i in crossing:
        t = d.date.iloc[i]
        if not events or (t-events[-1]).days >= separation: events.append(t)
    return pd.DatetimeIndex(events)

def onsets(d, quiet_days=7):
    """Identify alarm starts after seven complete quiet days; censor the first seven."""
    d = d.sort_values("date")
    a = d.alarm.to_numpy(dtype=int)
    return pd.DatetimeIndex(d.date.iloc[[i for i in range(quiet_days,len(a))
                                         if a[i]==1 and not a[i-quiet_days:i].any()]])

def plot_confusions(p, output):
    """Plot OOS 15%/30-,45-,60-day confusion matrices from tuned notebook 08b."""
    need = {"date","drop_pct","horizon","model","feature_group","y_true","y_pred"}
    if not need <= set(p): raise ValueError(f"Missing 08b fields: {sorted(need-set(p))}")
    d = p[(p.drop_pct==15)&(p.model=="XGBoost")&
          (p.feature_group=="Price+LPPLS")].copy()
    d["date"] = pd.to_datetime(d.date)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    rows=[]
    for ax,h in zip(axes,(30,45,60)):
        s=d[d.horizon==h].sort_values("date")
        if s.empty or s.date.duplicated().any(): raise ValueError(f"Invalid horizon: {h}")
        tn,fp,fn,tp=matrix_counts(s.y_true,s.y_pred)
        cm=np.array([[tn,fp],[fn,tp]])
        pct=100*cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
        ax.imshow(pct,vmin=0,vmax=100,cmap="Blues")
        for i in (0,1):
            for j in (0,1):
                ax.text(j,i,f"{cm[i,j]}\n({pct[i,j]:.1f}%)",ha="center",va="center",
                        color="white" if pct[i,j]>55 else "#263238",fontsize=10)
        ax.set_xticks([0,1],["No decline","Decline ≥15%"],fontsize=9)
        ax.set_yticks([0,1],["No decline","Decline ≥15%"],fontsize=9)
        ax.set_title(f"{h}-day horizon (n={len(s)})")
        ax.set_xlabel("Predicted class")
        rows.append(dict(horizon=h,n=len(s),TN=tn,FP=fp,FN=fn,TP=tp,
                         time_in_alarm=float(s.y_pred.mean())))
    axes[0].set_ylabel("Observed future decline")
    fig.suptitle("Out-of-sample confusion matrices: future BTC decline ≥15%")
    fig.savefig(output/"oos_confusion_future_drop_15pct.pdf",bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output/"oos_confusion_future_drop_15pct.csv",index=False)

def plot_alarm_comparison(p, reference, output):
    """Compare daily fixed OOS alarms and historical UNION intervals."""
    start,end=p.date.min(),p.date.max()
    prices=reference[reference.date.between(start,end+pd.Timedelta(days=60))]
    events=drawdown_events(reference)
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(12,6.3),sharex=True,
                              gridspec_kw={"height_ratios":[3,1.7]})
    ax.plot(prices.date,prices.price,color="black",lw=1,label="BTC-USD close")
    union=prices.bubble_union.fillna(0).astype(int).to_numpy()
    begins=np.flatnonzero((union==1)&~np.r_[False,union[:-1]])
    ends=np.flatnonzero((union==1)&~np.r_[union[1:],False])
    for i,(a,b) in enumerate(zip(begins,ends)):
        ax.axvspan(prices.date.iloc[a],prices.date.iloc[b],color="salmon",alpha=.24,
                   label="Retrospective UNION" if i==0 else None)
    for i,t in enumerate(events[(events>=start)&(events<=end)]):
        ax.axvline(t,color="firebrick",ls=":",lw=.75,
                   label="Drawdown onset (10%)" if i==0 else None)
    ax.set_ylabel("BTC-USD price (USD)")
    ax.legend(fontsize=8,ncol=3)
    labels=[]
    for i,(model,group,label) in enumerate(MODELS):
        s=p[(p.model==model)&(p.feature_group==group)].sort_values("date")
        if s.empty: continue
        ax2.fill_between(s.date,i-.35,i+.35,where=s.alarm.eq(1).to_numpy(),
                         step="mid",alpha=.85)
        labels.append((i,label))
    ax2.set_yticks([a for a,_ in labels],[b for _,b in labels],fontsize=9)
    ax2.set_ylim(len(MODELS)-.5,-.6)
    ax2.set_ylabel("OOS alarms")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.suptitle("Chronological OOS alarms and retrospective market events (2021–2024)")
    fig.autofmt_xdate(rotation=30)
    fig.savefig(output/"oos_alarm_timeline_bsadf_lppls_comparison.pdf",bbox_inches="tight")
    plt.close(fig)

def evaluate_event_metrics(p, reference, horizon=60):
    """Evaluate onset-level event recall and unconfirmed future-drop starts.

    Events are first drawdowns below 10% of the trailing 180-day high.
    Unconfirmed warnings use the distinct future-drop target in OOS rows.
    """
    events=drawdown_events(reference)
    start,end=p.date.min(),p.date.max()
    eligible=events[(events>=start+pd.Timedelta(days=horizon))&(events<=end)]
    rows=[]
    for (model,group),part in p.groupby(["model","feature_group"]):
        part=part.sort_values("date")
        starts=onsets(part)
        leads=[]
        for event in eligible:
            d=(event-starts).days
            good=d[(d>0)&(d<=horizon)]
            if len(good): leads.append(int(good.min()))
        not_confirmed=int((part.date.isin(starts)&part.target.eq(0)).sum())
        rows.append(dict(model=model,feature_group=group,n_events=len(eligible),
                         event_hits=len(leads),event_recall=len(leads)/len(eligible) if len(eligible) else np.nan,
                         median_lead_days=float(np.median(leads)) if leads else np.nan,
                         time_in_alarm=float(part.alarm.mean()),
                         unconfirmed_onsets_per_year=not_confirmed*365.25/len(part)))
    return pd.DataFrame(rows)

def plot_tradeoffs(metrics,output):
    """Render event capture and warning cost using preselected OOS alarms."""
    key=metrics.set_index(["model","feature_group"])
    for suffix,xfield,yfield,xlab,ylab,title in [
      ("oos_event_recall_vs_time_in_alarm","time_in_alarm","event_recall",
       "Time in alarm (% of OOS days)","Event recall (% of assessable onsets)",
       "Event recall vs. time in alarm"),
      ("oos_lead_time_vs_unconfirmed_alarms","unconfirmed_onsets_per_year","median_lead_days",
       "Unconfirmed warning starts per year","Median lead time (days)",
       "Warning lead time vs. unconfirmed alarms")]:
        fig,ax=plt.subplots(figsize=(8,5))
        for model,group,name in MODELS:
            if (model,group) not in key.index: continue
            r=key.loc[(model,group)]
            x,y=float(r[xfield]),float(r[yfield])
            if np.isnan(y): continue
            if suffix=="oos_event_recall_vs_time_in_alarm":x,y=100*x,100*y
            ax.scatter(x,y,s=70)
            ax.annotate(name,(x,y),xytext=(5,-16 if name=="Full hybrid" else 5),
                        textcoords="offset points",fontsize=8)
        ax.set(xlabel=xlab,ylabel=ylab,title=title)
        ax.grid(alpha=.25)
        fig.savefig(output/(suffix+".pdf"),bbox_inches="tight")
        plt.close(fig)

def main():
    """Generate the publication figures from authentic, precomputed OOS files."""
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-08b",type=Path,required=True)
    ap.add_argument("--pred-18",type=Path,required=True)
    ap.add_argument("--reference",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    pred08=pd.read_csv(a.pred_08b)
    pred18=check_predictions(pd.read_csv(a.pred_18))
    reference=pd.read_csv(a.reference,parse_dates=["date"])
    if reference.date.duplicated().any() or not {"price","bubble_union"}<=set(reference):
        raise ValueError("Reference must contain unique dates, prices and UNION states")
    plot_confusions(pred08,a.output)
    plot_alarm_comparison(pred18,reference,a.output)
    rows=evaluate_event_metrics(pred18,reference)
    rows.to_csv(a.output/"oos_event_tradeoff_metrics.csv",index=False)
    plot_tradeoffs(rows,a.output)
    print(rows.to_string(index=False))

if __name__=="__main__":
    main()
