# Reproducible LPPLS + machine learning research pipeline

The repository is designed to reuse **precomputed genuine LPPLS rolling fits**;
notebook 01 does not need to be executed in the normal workflow.

## Files required in the repository root

- `btc_usd_2015_2024.csv` — 3,653 rows with `Date, price` (real historical closes).
- `lppls_rolling_results.csv` — complete 107,240 LPPLS fits, including rejected fits.
- `lppls_rolling_results_z_etykietami_baniek.csv` — derived labeled fits for 04 (can be recreated quickly by 02, without refitting).
- `lppls_daily_validation.csv` — derived daily Confidence Indicator for 07 (can be recreated by 04).

**Do not** substitute a synthetic dataset from a unit-test or smoke-test folder.
The pipeline refuses unexpected counts, missing calendar dates and mismatched fit
aggregations. Note that the full fit count is specific to BTC 2015–2024.

## Installation

```bash
pip install -e ".[research,test]"
pytest -q
python scripts/audit_inputs.py
```

Install the additional `lppls` package only when you intend to perform new
model fittings or the optional multi-start test from notebook 11.

## Execution order

1. **Input audit**: `python scripts/audit_inputs.py` — no LPPLS fitting.
2. **02–04**: derived UNION labels and daily Confidence Indicator; 02/04 can be skipped if their validated CSVs are already supplied.
3. **07**: 9 forward-drop targets with fully observed horizons and label-availability dates.
4. **08b**: nested chronological tuning and test predictions for five models, three feature groups and nine target configurations. Computationally expensive.
5. **09, 10 (corrected), 10b, 11**: uncertainty, event warnings, operational threshold tuning and filter sensitivity.
6. **13**: sampled BSADF reference (not a full validated PSY implementation).
7. **14 → 15**: BSADF anomaly-state model and dependent warning evaluation.
8. **12**: final comparison tables. `16` is optional and requires **genuine** stock-index prices and a complete, independently fitted stock LPPLS dataset.

The deprecated notebooks 03, 08 (untuned) and the original 10 are **not** part
of publication runs. The primary ML workflow is 08b → 09/10b/12, and the
state-classification workflow is 13 → 14 → 15.

## What has actually been validated

- All supplied 107,240 BTC fits were audited and 1,518 passed the baseline filter.
- LPPLS summary features from 08b and 14 match on all 3,534 common dates.
- `04`, `07`, `13`, `15`, `10b`, `10 corrected`, `11` (without costly refits),
  and `12` executed successfully on the archived prices/outputs.
- Model 14 was retrained with the complete fit table, and 15 rerun on its new predictions.
- Existing 08b and 09 model predictions/results were **not retrained in this run**;
  08b daily features were reproduced exactly against archived 08b features.

## Outstanding scientific limitations

- 13 uses **sampled** BSADF with lag-0 ADF and a finite Monte Carlo sample. Its
  econometric calibration and alternative deterministic specifications still
  require an independent scientific implementation check.
- Full multi-initialization refitting in 11 requires installing `lppls` and
  explicitly enabling `RUN_REFIT_STABILITY`.
- Earlier examined 2021–2024 folds are chronological OOS but **not a previously
  unseen final test**; a precommitted, frozen holdout is needed for a stronger
  out-of-sample claim.
- The stock-index pilot is not an effective positive-class validation; 16
  now fails clearly when the test reference has only one class.
- The literature-review/novelty revision is manuscript work, not a Python test.

Notebook file names and data folder names are case-sensitive. Experiment
metadata/hashes are written to `input_audit.json`.