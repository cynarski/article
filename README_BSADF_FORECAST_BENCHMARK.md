# Future sampled-BSADF state onsets and BTC price-correction benchmarks

Notebooks 17 and 18 add prospective comparisons; they do not replace the current-state classification in notebook 14. They do not re-estimate 107,240 LPPLS windows.

## Required inputs in repository root

- 08b_wyniki_tuning/01_cechy_dzienne.csv: verified daily price/LPPLS features.
- 13_wyniki_bubble/13_bubble_reference_daily.csv: corrected historical sampled-BSADF reference.

## How to run

First: pip install -e '.[research,test]'
Then: PYTHONPATH=src pytest -q tests/test_event_benchmarks.py

Run notebook 17 for future sampled-BSADF onset within 30/60 daily observations; run notebook 18 for a future BTC >=10% decline within 60 daily observations. Both notebooks call scripts/run_statistical_benchmarks.py, report_statistical_benchmarks.py, and (for onsets) event_level_validation.py.

Both tasks use expanding year-based walk-forward (test 2021-2024), purging future labels at train/validation/test boundaries, validation-only threshold selection under a 30% time-in-alarm budget, common test dates across model variants, and as-of BSADF scores. The sample includes 18 distinct historical BSADF onsets, NOT hundreds of independent bubbles.

XGBoost settings are fixed across groups: 180 trees, maximum depth 3, learning rate 0.06, row and column sampling 0.8, minimum child weight 3, L2 regularization 5, seed 42. This differs from notebook 08b's nested tuning. Statistical BSADF-gap and LPPLS Confidence Indicator rules and AlwaysPositive are included as nontrained baselines. 600 repeated 60-day stratified moving-block bootstrap comparisons report uncertainty.

## Recomputed exploratory results (OOS 2021-2024)

| Task | OOS days | Positive days | XGB Price AP | XGB Price+BSADF AP | XGB Price+LPPLS+BSADF AP |
|---|---:|---:|---:|---:|---:|
| >=10% future decline within 60d | 1401 | 764 | 0.4830 | 0.5225 | 0.5488 |
| Future sampled-BSADF onset within 30d | 1236 | 134 | 0.2355 | 0.3487 | 0.3533 |
| Future sampled-BSADF onset within 60d | 1213 | 235 | 0.3081 | 0.3617 | 0.3624 |

The nontrained BSADF-gap rule reaches AP=0.6296 for 10%/60d corrections. Incremental LPPLS contribution above Price+BSADF for future onsets is 0.0046 (30d) and 0.0007 (60d); both block-bootstrap intervals include zero.

## Scientific limitations

The sampled-BSADF proxy is not the full PSY test. Future sampled-BSADF state onsets are not unambiguous fundamental bubbles, and price declines are not automatically bubble bursts. Repeated study of test years during project development makes results exploratory, not a virgin final holdout. All charts have English labels; output files are created only when the notebooks/scripts are executed.