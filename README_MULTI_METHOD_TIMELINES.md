# Multi-method OOS early-warning figures

This addition uses only saved out-of-sample forecasts, avoiding re-fitting
the 107,240 LPPLS rolling windows or reselecting thresholds on test labels.

## Seven series on identical OOS dates

| Family | Model/configuration | Protocol |
| --- | --- | --- |
| Structural rule | LPPLS Confidence Indicator | Previously calibrated, fixed-config comparison |
| Statistical rule | BSADF-gap | Previously calibrated, fixed-config comparison |
| Classical ML | Logistic Regression, Price+LPPLS | Chronologically tuned ML experiment |
| Decision tree | Decision Tree, Price+LPPLS | Chronologically tuned ML experiment |
| Boosting | XGBoost, Price+LPPLS | Chronologically tuned ML experiment |
| Boosting | LightGBM, Price+LPPLS | Chronologically tuned ML experiment |
| Boosting | CatBoost, Price+LPPLS | Chronologically tuned ML experiment |

All seven methods have the **same 1,401 OOS dates (2021-01-01 through
2024-11-01)** and the same binary target: a BTC decline of 10% or more
relative to the day of prediction within the following 60 days.

**Important:** these are two different earlier tuning procedures, although
their labels and OOS calendar match. The chart makes alarm coverage visible
but is not a controlled comparison under equal training budgets or equal
Time-in-Alarm constraints. UNION episode shading is *retrospective context*,
not a label used for the OOS future-decline outcome.

## Reproduce

```bash
python scripts/plot_oos_multi_method_timelines.py \
  --pred-08b 08b_wyniki_tuning/08b_04_predykcje_OOS.csv \
  --pred-18 18_wyniki_bsadf_future_drop/drop_10_h60/OOS_predictions.csv \
  --reference 13_wyniki_bubble/13_bubble_reference_daily.csv \
  --out 20_wyniki_multi_method_timelines
```

Produces `oos_seven_methods_compact.pdf` for the body of the paper,
`oos_seven_methods_price_panels.pdf` for an appendix, and CSVs for daily
alarm states and per-method counts.

## Publication-ready interpretation

The chronological tuned models spend a much larger share of their test
calendar in alarm than the LPPLS and BSADF rules. This finding concerns the
specific models and historical thresholds actually tested, not a universal
property of any model family. Accuracy, precision, sensitivity and raw
event recall must be interpreted alongside Time in Alarm.

The full LaTeX/Overleaf manuscript and pre-rendered vector figures are
delivered as a separate artifact. Neither external Python nor live data
access is needed for Overleaf compilation.
