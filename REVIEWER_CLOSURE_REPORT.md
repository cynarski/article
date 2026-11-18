# Reviewer comments 1–6: progress and scientific limits (2026-11-18)

This branch contains the historically reproducible sampled-BSADF Monte Carlo
fix and regression tests. A complete, executable notebook/results bundle was
prepared separately. **Do not reuse old notebook 13–15 model outputs after this
reference change.** Recompute the BSADF labels and all dependent models.

## Reproduced input data

- BTC daily closes, 2015–2024: **3,653**.
- Precomputed raw LPPLS windows: **107,240**; accepted: **1,518**.
- Dates with LPPLS fit coverage: **3,624**; positive signal dates: **556**.
- Corrected sampled-BSADF reference: **3,534** labeled days; **693** positives.
- XGBoost test dates (2021–24): **1,461** days; **195** positives.
- The initial LPPLS fitting notebook does **not** need to be rerun.

## Main findings after full local rerun of notebooks 13–15

- BSADF classification AP: XGBoost Price-only **0.647589**, XGBoost Price+LPPLS **0.653394**.
- Paired delta AP **+0.005805**, block-bootstrap 95% CI (60-day) **[-0.016074, +0.033051]**. There is no supported improvement of the hybrid.
- Price-only logistic regression AP: **0.716241**.
- Always-positive baseline AP: **0.133470**, F1 **0.235507**.
- 60 independent ADF(0) comparisons against statsmodels: max absolute statistic error **9.92e-11**.
- 252 comparisons at scheduled historical endpoints: Monte Carlo critical values were identical between historical prefix and full sample.
- LPPLS sensitivity (prespecified filters R2>0.80, 0.90, 0.95): XGBoost hybrid AP **0.655221**, **0.653394**, **0.664141**.
- Independent SciPy multistart stress: 60 stratified windows, 8 seeds/window (480 fits), 12 windows with varying acceptance outcome.
- Of 27 10b threshold selections: only **7** met the constraints; 18 had too few calibration events and 2 used an infeasible fallback.

## Unresolved scientific dependencies

1. No virgin 2025 test: authentic 2025 BTC closes and complete 2025 LPPLS fits were not available. Frozen BTC models and an OOS scorer were prepared, but no 2025 score was computed.
2. The sampled BSADF implementation is still **not** a complete PSY/GSADF calibration. Its grid and Monte Carlo size affect the reference; describe it as an approximation.
3. The multistart stress uses an independent SciPy optimization scheme; rerun the same original lppls optimizer if algorithm-specific stability is required.
4. The obsolete 1,262/1,696/338-vs-340 observation counts cannot be explained without the legacy row-level datasets. Supersede them rather than invent a reason.
5. Integrate literature comparisons and revised numbers into the manuscript. Do not assert statistically established superiority.

## Code requirements

Install research extras and run regression tests:

```bash
pip install -e '.[research,test]'
PYTHONPATH=.:src pytest -q tests
```

The full downloadable results bundle includes notebooks 04–17, advanced
validation scripts, CSV outputs, graphs, reproducibility reports and Polish
LaTeX sections. GitHub main does not yet include all generated experimental
artifacts, and the complete rerun was carried out outside GitHub.
