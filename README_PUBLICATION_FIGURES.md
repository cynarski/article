# OOS figures for manuscript Sections 5–6

Notebook **19_Publication_Confusion_Matrices_and_Warning_Plots.ipynb**
generates the publication figures in English using existing OOS predictions.
No LPPLS refitting, model retraining or test-set threshold optimization occurs.

## Inputs

1. `08b_wyniki_tuning/08b_04_predykcje_OOS.csv` – chronological
   (tuned) 2021–2024 predictions for 15% future BTC price declines
   at 30-, 45- and 60-day horizons.
2. `18_wyniki_bsadf_future_drop/drop_10_h60/OOS_predictions.csv` –
   10% future decline in 60 days, fixed-configuration feature-family
   benchmark (including BSADF/LPPLS rules), also OOS.
3. `13_wyniki_bubble/13_bubble_reference_daily.csv` – authentic BTC
   price history with retrospective literature UNION intervals.

Run the notebooks 08b, 13 and 18 before notebook 19. Alternatively:

```sh
python scripts/plot_oos_publication_figures.py \
  --pred-08b 08b_wyniki_tuning/08b_04_predykcje_OOS.csv \
  --pred-18 18_wyniki_bsadf_future_drop/drop_10_h60/OOS_predictions.csv \
  --reference 13_wyniki_bubble/13_bubble_reference_daily.csv \
  --output 19_wyniki_publication_figures
```

Output:
- `oos_confusion_future_drop_15pct.pdf` (three horizons, counts and row %)
- `oos_alarm_timeline_bsadf_lppls_comparison.pdf`
- `oos_event_recall_vs_time_in_alarm.pdf`
- `oos_lead_time_vs_unconfirmed_alarms.pdf`
- `oos_confusion_future_drop_15pct.csv`
- `oos_event_tradeoff_metrics.csv`

### Scientific caveats

The 15% matrices come from the tuned **08b** experiment. The early-warning
comparison uses the distinct fixed-hyperparameter **18** experiment.
Results must not be conflated, including their distinct test horizons.

Event recall counts new alarms preceding first crossings of a 10% decline
from the *previous* 180-day price maximum, enforcing 60 days' separation.
It uses a 60-day prewarning window and censors events without 60 days
of earlier OOS alarm data. Alarm starts require seven previous quiet days.
A warning is **unconfirmed** if its day does not lead to a >=10% decline
relative to *that day's price* over 60 days. The two event definitions
are not equivalent. UNION is a retrospective literature overlay, not
ground truth for future decline forecasting. Alarm thresholds were
selected before the OOS periods.

The manuscript figures based on the actual CSVs are also provided in a
standalone Overleaf project alongside the Polish Sections 5 and 6.
