# SARIMAX component

The default `revised` profile uses the strengthened cyclist model and the
pedestrian model supported by the follow-up investigation. Motorists retain the
original model. The `original` profile preserves the initial three-candidate
search. Original outputs in `OUTPUT/sarimax/full/` are preserved.

## Run from the repository root

Python 3.13 is the verified environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s SCRIPTS -p 'test_*.py'
.venv/bin/python -B SCRIPTS/sarimax.py --mode smoke
.venv/bin/python -B SCRIPTS/sarimax.py --mode validate --jobs 3
.venv/bin/python -B SCRIPTS/sarimax.py --mode full --jobs 3
```

Each command writes to its own directory under `OUTPUT/sarimax/revised/`.
An existing run is never overwritten: specify a fresh `--output` directory for
reruns, for example:

```sh
.venv/bin/python -B SCRIPTS/sarimax.py --profile revised --mode full --jobs 3 --output OUTPUT/sarimax/revised/rerun1
.venv/bin/python -B SCRIPTS/sarimax.py --profile original --mode full --jobs 3 --output OUTPUT/sarimax/original-rerun
```

`--maxiter` defaults to 200. `--input` overrides the processed-data path.
`--jobs` allows 1–3 parallel CV fits, defaulting to 1. Final fitting is serial.
The temporary session environment is `/private/tmp/ds4002-sarimax-venv`; create
`.venv` above for a persistent environment.

Modes:

- `smoke`: one first-fold fit; cyclists for revised, pedestrians for original.
- `benchmark`: first-fold fits for all revised group configurations, or the three
  original candidates for pedestrians.
- `validate`: all pretest CV fits and scores; no final fits or test forecasts.
- `full`: validation followed by three final fits and 36 forecast rows.

## Data and approved dates

Input is `DATA/weekly_injuries_processed.csv`; the identical OUTPUT copy is not
used. Preprocessing, data, and teammate code are not modified. The targets count
injured people, as established by `preprocess.py` and its aggregation consistency
log. There are 512 complete Monday–Sunday weeks labelled by `week_ending`
(`W-SUN`), 2016-03-13–2025-12-28. Partial edge weeks were already removed.

The existing features are reused in the order `holiday_week, covid_period`.
Holiday weeks include U.S. federal observed holidays. COVID weeks overlap the
existing inclusive interval 2020-03-01–2021-06-30. Only calendar-known features
are supplied for forecasts.

The user-approved literal 280-week initial window is 2016-03-13–2021-07-18.
The first validation window is 2021-07-25–2021-10-10. Training expands by 12
weeks per fold, with 18 full validation windows; the last is
2025-06-22–2025-09-07. Four remaining pretest weeks do not form another full fold
but are included in final fitting. Final fitting uses all 500 observations
through 2025-10-05. Each fold forecasts its next 12 weeks in one call, with no
updates from validation targets.

The original 12-week holdout dates are 2025-10-12, 2025-10-19, 2025-10-26,
2025-11-02, 2025-11-09, 2025-11-16, 2025-11-23, 2025-11-30, 2025-12-07,
2025-12-14, 2025-12-21 and 2025-12-28. They were protected during the original
search but have since been examined. Revised results on these weeks are
**retrospective exploratory results, not independent confirmation**.

## Revised configurations and adoption evidence

| Group | Order | Seasonal order | Trend |
|---|---|---|---|
| Pedestrians | (1,0,1) | (0,1,1,52) | None |
| Cyclists | (0,0,1) | (0,1,1,52) | None |
| Motorists | (0,1,1) | (0,1,0,52) | Constant |

The revised profile fixes these configurations; it validates them rather than
rerunning a parameter search. All 18 folds must converge before a final export.
The original search and the new investigations are archived separately.

Cyclist adoption is based on an 18-fold mean MAE of 13.626344 versus 16.024800
for the original model; raw 80% interval coverage was 84.3% over 216 validation
observations. Removing the constant alone made little difference. Adding a
seasonal moving-average term and reducing differencing provided the larger gain.

For pedestrians, the first smoothing alternative improved MAE to 24.944657 but
had residual correlation and missed the coverage goal on the already-seen
holdout. A further pretest-only experiment added an autoregressive term. Its
18-fold mean MAE was 22.835923 versus the original 28.360644, with 91.2% coverage.
All 18 fits converged without warnings, and none of their residual Ljung–Box
checks rejected at the 5% level at either tested lag (12,52). These nominal tests
support the choice but do not prove independence or guarantee future coverage.
Its mean interval width (92.11) was smaller than the earlier revision's (95.95).

A separate interval check tested widening using only errors from earlier
validation folds, starting after six folds (72 observations). For the earlier
pedestrian model, widening eventually became unnecessary as coverage improved.
No arbitrary multiplier was chosen to cover the known holdout misses. **Exported
intervals remain raw nominal 80% model intervals.** The diagnostic widening rule
is not applied to final forecasts and provides no time-series coverage guarantee.
Reproduce that diagnostic with:

```sh
.venv/bin/python -B SCRIPTS/check_intervals.py
```

The motorist model stays unchanged. Its small exploratory validation gain from
removing the constant was insufficient reason to change the approved model.
All model choices were fixed before regenerating final forecasts.

## Original search

With `--profile original`, every group uses the same declared candidate grid:

1. `(1,0,0)(0,0,0,52)`, constant trend.
2. `(1,0,0)(1,0,0,52)`, constant trend.
3. `(0,1,1)(0,1,0,52)`, constant trend.

This is 162 CV fits plus three final fits; revised full mode has 54 CV fits plus
three final fits. Only candidates converged on every fold are eligible.
Minimum mean fold MAE selects the original winner; exact ties use declaration
order. Partial-fold averages are never eligible scores.

Both profiles enforce stationarity and invertibility and use deterministic
L-BFGS fitting with an explicit iteration limit and random seed 4002.
Failures, initialization warnings, convergence, timing, residual summaries, and
Ljung–Box statistics at lags 12 and 52 are saved. Final nonconvergence prevents
forecast export. Predictions and intervals are never clipped.

## Outputs and evaluation interface

Revised run outputs are under `OUTPUT/sarimax/revised/<mode>/`:

- `run.json`: status, profile, complete configurations, versions, input hash and dates.
- `diagnostics.json`: fold and final fit diagnostics.
- `fold_scores.csv`, `candidate_scores.csv`: MAE, baseline MAE, coverage and width.
- `validation_forecasts.csv`: pretest out-of-fold actuals, point forecasts and intervals.
- `best_orders.json`: validated configurations, written before final test evaluation.
- `forecasts.csv`: full mode only; 36 rows with `week_ending`, `predicted`,
  `lower_80`, `upper_80`, `actual`, `seasonal_naive`, and `group`.

Check `run.json` has `status: complete` before consuming exports. Input SHA-256
is checked after execution. The 52-week seasonal-naive values and unchanged final
schema support the evaluation teammate's MAE, percent improvement and interval
coverage calculations. No separate production evaluation module is introduced.

Archived investigation evidence is in `OUTPUT/sarimax/investigation/`: candidate
scores and diagnostics, pedestrian validation forecasts, the interval history
checks, the follow-up plan and scores, and `decisions.json` recording adoption
criteria, configurations, package versions, and protected original-output hashes.

The original verified run in `OUTPUT/sarimax/full/` had mean validation MAEs
28.360644, 16.024800 and 65.497974, respectively. All original fits converged.
Its residual diagnostics showed remaining correlation. Those results are retained
for comparison rather than replaced by the revised study.

Coverage of at least 80% on 12 weeks requires at least 10 observations inside
the intervals. Neither optimizer convergence nor historical validation establishes
that project goals will hold in a fresh period; independent confirmation is
still needed, especially for cyclist residual behavior.
