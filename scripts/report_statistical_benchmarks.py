"""English plots and paired block-bootstrap contrasts for forecast benchmarks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


def resample_by_year(d: pd.DataFrame, block: int, rng) -> np.ndarray:
    """Sample moving blocks without crossing yearly evaluation boundaries."""
    chosen=[]
    for _,part in d.groupby(d.date.dt.year,sort=True):
        idx=part.index.to_numpy()
        n=len(idx)
        draw=[]
        while len(draw)<n:
            start=int(rng.integers(0,n))
            draw.extend(idx[start:min(start+block,n)].tolist())
        chosen.extend(draw[:n])
    return np.asarray(chosen,dtype=int)


def paired_delta_ap(preds: pd.DataFrame, *, repeats: int=600,
                    block: int=60, seed: int=42) -> pd.DataFrame:
    """Estimate paired AP differences using year-stratified block bootstrap."""
    p=preds.loc[preds.model.eq("XGBoost")].copy()
    wide=p.pivot(index="date",columns="feature_group",values="score")
    d=wide.join(p.groupby("date").target.first()).reset_index().sort_values("date").reset_index(drop=True)
    pairs=[("Price+BSADF","Price-only","BSADF-added-to-price"),
           ("Price+LPPLS","Price-only","LPPLS-added-to-price"),
           ("Price+LPPLS+BSADF","Price+BSADF","LPPLS-added-to-price+BSADF")]
    rng=np.random.default_rng(seed)
    indexes=[resample_by_year(d,block,rng) for _ in range(repeats)]
    rows=[]
    for high,low,label in pairs:
        point=average_precision_score(d.target,d[high])-average_precision_score(d.target,d[low])
        values=[]
        for idx in indexes:
            y=d.target.iloc[idx]
            if y.nunique()<2:continue
            values.append(average_precision_score(y,d[high].iloc[idx]) -
                          average_precision_score(y,d[low].iloc[idx]))
        rows.append({"contrast":label,"high":high,"low":low,
                     "point_delta_AP":point,"bootstrap_low_95":np.quantile(values,.025),
                     "bootstrap_high_95":np.quantile(values,.975),
                     "block_size":block,"repeats_valid":len(values)})
    return pd.DataFrame(rows)


def make_figures(folder: Path, label: str, *, block: int=60):
    """Save figure PDFs and a paired-difference CSV for a completed task."""
    summary=pd.read_csv(folder/"OOS_summary.csv")
    preds=pd.read_csv(folder/"OOS_predictions.csv",parse_dates=["date"])
    delta=paired_delta_ap(preds,block=block)
    delta.to_csv(folder/"paired_deltaAP.csv",index=False)
    benchmark=summary.loc[summary.model.eq("AlwaysPositive"),"AP"].iloc[0]
    data=summary.loc[summary.model.eq("XGBoost")].sort_values("AP")
    fig,ax=plt.subplots(figsize=(10,4.7))
    ax.barh(data.feature_group,data.AP,color=plt.cm.viridis(np.linspace(.25,.8,len(data))))
    ax.axvline(benchmark,color="black",ls="--",label="Positive-class prevalence")
    ax.set(xlabel="Average Precision",title=label)
    ax.legend(loc="lower right")
    ax.set_xlim(0,max(.7,data.AP.max()*1.1))
    fig.tight_layout()
    fig.savefig(folder/"AP_benchmark_EN.pdf")
    fig.savefig(folder/"AP_benchmark_EN.png",dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.3))
    y=np.arange(len(delta))
    ax.errorbar(delta.point_delta_AP,y,
                xerr=[delta.point_delta_AP-delta.bootstrap_low_95,
                      delta.bootstrap_high_95-delta.point_delta_AP],
                fmt="o",color="navy",capsize=4)
    ax.axvline(0,color="firebrick",ls="--")
    ax.set_yticks(y,delta.contrast)
    ax.set(xlabel="Difference in Average Precision (paired block-bootstrap 95% CI)",
           title="Incremental value of feature families")
    fig.tight_layout()
    fig.savefig(folder/"DeltaAP_benchmark_EN.pdf")
    fig.savefig(folder/"DeltaAP_benchmark_EN.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--block",type=int,default=60)
    args=parser.parse_args()
    make_figures(args.folder,args.label,block=args.block)
    print(pd.read_csv(args.folder/"paired_deltaAP.csv").to_string(index=False))
