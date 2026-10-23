# SARIMAX component

Run from the repository root with Python 3.13 (the verified environment):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s SCRIPTS -p 'test_*.py'
.venv/bin/python SCRIPTS/sarimax.py --mode smoke
.venv/bin/python SCRIPTS/sarimax.py --mode benchmark
.venv/bin/python SCRIPTS/sarimax.py --mode full
```

The smoke mode fits one candidate for pedestrians on the first fold; benchmark
fits all three declared candidates on that fold. Neither accesses test targets.
Full mode runs all three groups, 18 folds and three candidates (162 CV fits),
then three final fits. `--maxiter` defaults to 200; `--input` and `--output`
override paths. The candidate list is configurable in `SCRIPTS/sarimax.py`;
freeze it before full CV and never revise it using test results.

## Data and approved splits

Input is `DATA/weekly_injuries_processed.csv`. The identical OUTPUT copy is not
used. No preprocessing or input files are modified. Targets count injured people,
as established by `preprocess.py` and its aggregation consistency log. Monday–Sunday
weeks use `week_ending` Sunday labels (`W-SUN`); 512 observations span
2016-03-13 through 2025-12-28. Existing federal holiday (including observed holiday)
and COVID-overlap indicators are reused in the order `holiday_week, covid_period`.
The existing COVID interval is 2020-03-01 through 2021-06-30 inclusive.

The approved literal initial window has 280 weeks, 2016-03-13–2021-07-18.
Validation starts 2021-07-25–2021-10-10; training expands by 12 weeks per fold.
There are 18 complete folds, ending with validation 2025-06-22–2025-09-07.
Four remaining pretest weeks do not form a complete validation fold. They enter
the final refit, which uses all 500 observations through 2025-10-05.
The untouched test dates are 2025-10-12, 2025-10-19, 2025-10-26, 2025-11-02,
2025-11-09, 2025-11-16, 2025-11-23, 2025-11-30, 2025-12-07, 2025-12-14,
2025-12-21, and 2025-12-28. The first validation window starts after the COVID
indicator ends; this exact interpretation was confirmed by the user.

## Search and outputs

Declared `(order, seasonal_order)` candidates, in tie-break order:

1. `(1,0,0), (0,0,0,52)`
2. `(1,0,0), (1,0,0,52)`
3. `(0,1,1), (0,1,0,52)`

All include a constant trend and both exogenous features, with stationarity and
invertibility enforced. Statsmodels' default deterministic L-BFGS optimizer is
used with an explicit iteration limit. The random seed is 4002.
Each fold makes one 12-step forecast using calendar-known exogenous values.
Only candidates with successful, converged fits on all 18 folds are eligible;
mean validation MAE selects the winner and exact ties use declaration order.
Partial-fold averages are not selection scores. Final nonconvergence stops export.
Warnings, residual mean/std and Ljung–Box results at lags 12 and 52 are recorded
without changing the selection rule. No forecasts or interval bounds are clipped.

Outputs are separated into `OUTPUT/sarimax/smoke/`, `benchmark/`, and `full/`:

- `run.json`: exact configuration, package versions, input hash and split dates.
- `diagnostics.json`: failures, convergence, warnings, timing and residual diagnostics.
- `fold_scores.csv`, `candidate_scores.csv`: CV results (empty in benchmark modes).
- `best_orders.json`: each group's chosen orders and validation MAE (full mode).
- `forecasts.csv`: 36 rows, with `week_ending`, `predicted`, `lower_80`,
  `upper_80`, `actual`, `seasonal_naive`, and `group` (full mode).

Full outputs are written only after every final fit succeeds. Use a fresh
`--output` directory for each rerun to avoid confusing old exports with failures.
For full runs, check that `run.json` has `status: complete` before consuming
forecasts; a failed rerun may leave exports from a previous run in that directory.
The evaluation teammate can use `seasonal_naive` (52-week lag), actuals, and raw
80% intervals directly. This component does not duplicate final evaluation.
Coverage of at least 80% requires 10 of 12 test observations inside the intervals;
performance goals must be computed separately and are not assumed here.

## Verified run

The completed run in `OUTPUT/sarimax/full/` evaluated all 162 CV fits and
three final fits: all converged, with no fitting warnings or failures. Every group
selected `(0,1,1)(0,1,0,52)`. Mean validation MAEs were 28.360644 for pedestrians,
16.024800 for cyclists, and 65.497974 for motorists. The final export has 36 rows,
aligned test dates, and finite, ordered intervals; no point forecasts or lower
bounds were negative. Input SHA-256 was unchanged, and all six unit tests passed.

Final residual Ljung–Box p-values at lag 52 are below 0.001 for all three groups,
indicating remaining autocorrelation despite optimizer convergence. Interpret
nominal prediction intervals accordingly. Test MAE improvement and empirical
coverage remain for the evaluation teammate; these goals have not been asserted.

For this session, the dependency installation and execution used the temporary
environment `/private/tmp/ds4002-sarimax-venv`; use the setup commands above for
a persistent environment. Exact package versions are recorded in `full/run.json`.
