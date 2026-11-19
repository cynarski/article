"""Out-of-sample BSADF, LPPLS and ML benchmarks for future events.

Run after notebooks 07/08b and 13; raw LPPLS re-estimation is not required.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lppls_research.event_benchmarks import (
    bsadf_asof_features, future_onset_labels, purged_year_split, select_alarm_threshold
)

PRICE = ["ret_1d", "momentum_7d", "momentum_14d", "momentum_30d", "momentum_60d",
         "volatility_7d", "volatility_30d", "volatility_60d",
         "drawdown_30d", "drawdown_90d"]
LPPLS = ["lppls_ci", "lppls_n_accepted", "lppls_m", "lppls_w", "lppls_tgap",
         "lppls_r2", "lppls_B", "lppls_C", "lppls_window_n", "lppls_tgap_std"]
BSADF = ["bsadf_gap", "bsadf_age_days", "bsadf_gap_delta"]
FEATURE_GROUPS = {
    "Price-only": PRICE,
    "LPPLS-only": LPPLS,
    "BSADF-only": BSADF,
    "Price+LPPLS": PRICE + LPPLS,
    "Price+BSADF": PRICE + BSADF,
    "Price+LPPLS+BSADF": PRICE + LPPLS + BSADF,
}


def prepare_data(features_csv: Path, reference_csv: Path, task: str,
                 horizon: int, drop: int) -> pd.DataFrame:
    """Merge complete daily LPPLS features and causal BSADF reference signals."""
    features = pd.read_csv(features_csv, parse_dates=["date"])
    ref = pd.read_csv(reference_csv, parse_dates=["date", "reference_available_at"])
    if features.date.duplicated().any() or ref.date.duplicated().any():
        raise ValueError("Date must be unique in each source")
    d = features.merge(bsadf_asof_features(ref), on="date",
                       validate="one_to_one", how="left").merge(
        ref[["date", "bubble_state_gsadf", "reference_available_at"]],
        on="date", validate="one_to_one", how="left"
    ).sort_values("date").reset_index(drop=True)
    known = d.reference_available_at.notna()
    if (d.loc[known, "reference_available_at"] > d.loc[known, "date"]).any():
        raise ValueError("BSADF signal not known on decision date")
    if task == "future_drop":
        label = f"drop_{drop}_h{horizon}"
        label_end = f"label_end_h{horizon}"
        if label not in d or label_end not in d:
            raise ValueError(f"Missing future-drop columns: {label}, {label_end}")
        d["target"] = pd.to_numeric(d[label], errors="coerce")
        d["label_end"] = pd.to_datetime(d[label_end])
    elif task == "future_onset":
        labels = future_onset_labels(ref.date, ref.bubble_state_gsadf, horizon)
        labels = labels.rename(columns={"onset": "target"})
        d = d.merge(labels[["date", "eligible", "target", "label_end"]],
                    on="date", validate="one_to_one", how="left")
        d = d.loc[d.eligible].copy()
    else:
        raise ValueError("task must be future_drop or future_onset")
    d = d.loc[d.target.notna() & d.bsadf_gap.notna() &
              d.label_end.notna()].copy()
    d["target"] = d.target.astype(int)
    return d.sort_values("date").reset_index(drop=True)


def fit_model(name: str, train: pd.DataFrame, cols: list[str]):
    """Fit a model with fixed parameters shared across feature groups."""
    if name == "XGBoost":
        clf = XGBClassifier(n_estimators=180, max_depth=3, learning_rate=.06,
                            subsample=.8, colsample_bytree=.8, min_child_weight=3,
                            reg_lambda=5, tree_method="hist", n_jobs=2,
                            random_state=42, eval_metric="logloss")
        model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0), clf)
    elif name == "LogisticRegression":
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(SimpleImputer(strategy="constant", fill_value=0),
                              StandardScaler(), LogisticRegression(
                                  C=1., max_iter=2000, random_state=42))
    else:
        raise ValueError(name)
    model.fit(train[cols], train.target)
    return model


def classification_stats(y, scores, alarm) -> dict:
    """Aggregate all OOS dates without averaging yearly PR-AUC."""
    y, scores, alarm = np.asarray(y, dtype=int), np.asarray(scores, dtype=float), np.asarray(alarm, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, alarm, labels=[0, 1]).ravel()
    return {"n":len(y), "n_positive":int(y.sum()),
            "AP":float(average_precision_score(y, scores)) if len(np.unique(y))==2 else float("nan"),
            "ROC_AUC":float(roc_auc_score(y, scores)) if len(np.unique(y))==2 else float("nan"),
            "F1":float(f1_score(y,alarm,zero_division=0)),
            "recall":tp/(tp+fn) if tp+fn else float("nan"),
            "specificity":tn/(tn+fp) if tn+fp else float("nan"),
            "time_in_alarm":float(alarm.mean()),
            "TP":int(tp), "FP":int(fp), "FN":int(fn), "TN":int(tn)}


def run_experiment(features_csv: Path, reference_csv: Path, out: Path,
                   *, task: str, horizon: int, drop: int=10,
                   years: tuple[int,...]=(2021,2022,2023,2024),
                   budget: float=.30) -> dict:
    """Run expanding-year, purged walk-forward on identical test dates."""
    out.mkdir(parents=True, exist_ok=True)
    d = prepare_data(features_csv, reference_csv, task, horizon, drop)
    d.to_csv(out/"prepared_daily_data.csv", index=False)
    configs = [("XGBoost",group,cols) for group,cols in FEATURE_GROUPS.items()]
    configs += [("LogisticRegression", "Price-only", PRICE),
                ("StatisticalRule", "BSADF-gap", ["bsadf_gap"]),
                ("StatisticalRule", "LPPLS-CI", ["lppls_ci"]),
                ("AlwaysPositive", "Reference", [])]
    predictions, fold_metrics, skipped = [], [], []
    for year in years:
        tr,va,te = purged_year_split(d,year)
        if any(part.empty for part in (tr,va,te)):
            skipped.append({"year":year, "reason":"empty_fold"})
            continue
        if tr.target.nunique()<2:
            skipped.append({"year":year,"reason":"single_class_training"})
            continue
        for name,group,cols in configs:
            if name in ("XGBoost","LogisticRegression"):
                model = fit_model(name,tr,cols)
                val_scores = model.predict_proba(va[cols])[:,1]
                test_scores = model.predict_proba(te[cols])[:,1]
            elif name=="StatisticalRule":
                val_scores = va[cols[0]].fillna(0).to_numpy(float)
                test_scores = te[cols[0]].fillna(0).to_numpy(float)
            else:
                val_scores = np.ones(len(va),dtype=float)
                test_scores = np.ones(len(te),dtype=float)
            if name=="AlwaysPositive":
                threshold,status = float("-inf"),"fixed_always_positive"
            elif name=="StatisticalRule":
                grid=np.unique(np.quantile(val_scores,np.linspace(0,1,101)))
                threshold,status=select_alarm_threshold(
                    va.target,val_scores,max_time_in_alarm=budget,thresholds=grid)
            else:
                threshold,status=select_alarm_threshold(
                    va.target,val_scores,max_time_in_alarm=budget)
            alarm=(test_scores>=threshold).astype(int)
            row=te[["date","target"]].copy()
            row["year"]=year
            row["model"]=name
            row["feature_group"]=group
            row["score"]=test_scores
            row["alarm"]=alarm
            row["threshold"]=threshold
            row["threshold_status"]=status
            predictions.append(row)
            fold_metrics.append({"year":year,"model":name,"feature_group":group,
                                 "threshold":threshold,"threshold_status":status,
                                 "n_train":len(tr),"n_validation":len(va),
                                 **classification_stats(te.target,test_scores,alarm)})
    if not predictions:
        raise RuntimeError("No evaluable walk-forward folds")
    pred=pd.concat(predictions,ignore_index=True)
    if pred.duplicated(["date","model","feature_group"]).any():
        raise ValueError("Duplicate model prediction date")
    pred.to_csv(out/"OOS_predictions.csv",index=False)
    pd.DataFrame(fold_metrics).to_csv(out/"fold_metrics.csv",index=False)
    summary=pd.DataFrame([{"model":name,"feature_group":group,
                           **classification_stats(p.target,p.score,p.alarm)}
                          for (name,group),p in pred.groupby(["model","feature_group"])])
    summary.to_csv(out/"OOS_summary.csv",index=False)
    meta={"task":task,"horizon":horizon,"drop_pct":drop,
          "test_years":list(years),"max_time_in_alarm_validation":budget,
          "n_eligible_dates":len(d),"overall_positive_dates":int(d.target.sum()),
          "skipped_years":skipped,
          "note":"Fixed-parameter comparison; sampled BSADF state is a statistical proxy. "
                 "Not the same model-tuning experiment as notebook 08b."}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return {"summary":summary,"predictions":pred,"metadata":meta}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",type=Path,default=ROOT/"08b_wyniki_tuning/01_cechy_dzienne.csv")
    p.add_argument("--reference",type=Path,default=ROOT/"13_wyniki_bubble/13_bubble_reference_daily.csv")
    p.add_argument("--out",type=Path,default=ROOT/"17_18_wyniki_porownanie")
    p.add_argument("--task",required=True,choices=["future_drop","future_onset"])
    p.add_argument("--horizon",type=int,default=60)
    p.add_argument("--drop",type=int,default=10)
    a=p.parse_args()
    result=run_experiment(a.features,a.reference,a.out,task=a.task,horizon=a.horizon,drop=a.drop)
    print(result["summary"].to_string(index=False))
