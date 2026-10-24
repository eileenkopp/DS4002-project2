# Revised SARIMAX results

These are exploratory follow-up results. The 2025 holdout had already been examined; this is not independent confirmation.
Model choices were fixed using pretest evidence before final forecasts were regenerated. Raw nominal 80% intervals are used, without clipping or post-test widening.

| Group | Original CV MAE | Revised CV MAE | CV coverage | Original test MAE | Revised test MAE | Baseline improvement | Test coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| pedestrians | 28.36 | 22.84 | 91.2% | 30.84 | 19.58 | 30.08% | 12/12 (100.0%) |
| cyclists | 16.02 | 13.63 | 84.3% | 19.73 | 12.71 | 14.32% | 11/12 (91.7%) |
| motorists | 65.50 | 65.50 | 98.1% | 42.03 | 42.03 | 17.17% | 12/12 (100.0%) |

## Adopted models

- Pedestrians: (1,0,1)(0,1,1,52), no constant. The follow-up AR term improved pretest accuracy and residual diagnostics.
- Cyclists: (0,0,1)(0,1,1,52), no constant. Residual correlation remains a limitation.
- Motorists: unchanged (0,1,1)(0,1,0,52), constant. Regenerated forecasts match the original.

## Verification

- All 54 CV fits and three final fits converged.
- Validation export: 648 rows, all pretest; final export: 36 rows with aligned dates and numeric ordered intervals.
- Processed input SHA-256 and all original full-output hashes are unchanged.
- Fitting warnings logged: 1. See full/diagnostics.json.
- No commits were made. Pre-existing preprocessing-log edits were left alone.

## Files

- full/forecasts.csv: revised forecast export in the existing evaluation schema.
- full/validation_forecasts.csv: out-of-fold predictions and raw intervals.
- full/comparison.csv: full-precision descriptive comparison generated during verification.
- full/run.json, best_orders.json and diagnostics.json: configurations, versions and fit evidence.
- ../investigation/: archived candidate comparison, follow-up plan, scores, interval checks and adoption decisions.

Dates remain unchanged: initial training 2016-03-13–2021-07-18, 18 CV windows through 2025-09-07, final training through 2025-10-05, retrospective test 2025-10-12–2025-12-28.
Further independent evaluation is still required. High empirical coverage on a short period is not a guarantee of calibrated future intervals.
