"""Episode-level assessment for future sampled-BSADF onset warnings."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def score_events(predictions: pd.DataFrame, reference: pd.DataFrame,
                 horizon: int, *, cooldown: int=7) -> pd.DataFrame:
    """Count evaluable events, new alarms, lead time and unconfirmed starts.

    Only historical onsets with a complete OOS lookback are included; the
    labels remain a statistical sampled-BSADF proxy. Alarm starts following a
    gap are counted as new unless they fall inside the configured cooldown.
    """
    r=reference.sort_values("date").copy()
    r["date"]=pd.to_datetime(r.date)
    state=r.bubble_state_gsadf.astype("Float64")
    starts_state=state.eq(1)&state.shift(1).eq(0)
    event_dates=set(r.loc[starts_state.fillna(False),"date"])
    rows=[]
    for (model,group),p in predictions.groupby(["model","feature_group"]):
        p=p.sort_values("date").reset_index(drop=True)
        consecutive=p.date.diff().eq(pd.Timedelta(days=1))
        started=p.alarm.eq(1)&(~consecutive|p.alarm.shift(fill_value=0).eq(0))
        alarms=[]
        for date in p.loc[started,"date"]:
            if not alarms or (date-alarms[-1]).days>=cooldown:
                alarms.append(date)
        left,right=p.date.min(),p.date.max()
        examined=[date for date in sorted(event_dates)
                  if left+pd.Timedelta(days=horizon)<=date<=right]
        hit_any=0
        hit_new=0
        lead=[]
        for event in examined:
            previous=p[(p.date<event)&(p.date>=event-pd.Timedelta(days=horizon))]
            if previous.alarm.eq(1).any():
                hit_any+=1
            warnings=[x for x in alarms if event-pd.Timedelta(days=horizon)<=x<event]
            if warnings:
                hit_new+=1
                lead.append((event-warnings[0]).days)
        daily=p.set_index("date").target
        available_starts=[dt for dt in alarms if dt in daily.index and pd.notna(daily.loc[dt])]
        unconfirmed=sum(int(daily.loc[dt]==0) for dt in available_starts)
        years=(right-left).days/365.25+1/365.25
        rows.append({"model":model,"feature_group":group,
                     "n_events_evaluable":len(examined),
                     "event_hits_any_alarm":hit_any,"event_hits_new_alarm":hit_new,
                     "event_recall_any":hit_any/len(examined) if examined else np.nan,
                     "event_recall_new_onset":hit_new/len(examined) if examined else np.nan,
                     "median_lead_days":np.median(lead) if lead else np.nan,
                     "n_warning_starts_assessable":len(available_starts),
                     "unconfirmed_warning_starts":unconfirmed,
                     "unconfirmed_warnings_per_year":unconfirmed/years,
                     "time_in_alarm":p.alarm.mean()})
    return pd.DataFrame(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--horizon",type=int,default=60)
    args=parser.parse_args()
    pred=pd.read_csv(args.folder/"OOS_predictions.csv",parse_dates=["date"])
    ref=pd.read_csv(args.reference,parse_dates=["date"])
    result=score_events(pred,ref,args.horizon)
    result.to_csv(args.folder/"event_level_summary.csv",index=False)
    print(result.to_string(index=False))
