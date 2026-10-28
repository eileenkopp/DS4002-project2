"""
sarimax.py - Part 2: SARIMAX forecasts of weekly NYC injury counts

Purpose : Fit a SARIMAX model for each road-user group, test it with rolling-origin
          cross-validation, and forecast the final 12 weeks of 2025 with 80%
          intervals, compared against a seasonal-naive baseline.

Run order: run after preprocess.py
Input   : DATA/weekly_injuries_processed.csv
Output  : folder set by --output (never overwrites an existing run)

How to run: python SCRIPTS/sarimax.py --mode full --jobs 3 --output OUTPUT/my_run
Modes   : smoke:    (default) one first-fold fit; writes smoke_<group>_candidate_0.csv
          validate: all rolling-origin CV fits and scores; no final fits or forecasts.csv
          full:     validation, then final fits and forecasts.csv (reproduces our results)
Notes: Without options, the script runs only a quick test (--mode smoke).
       See SCRIPTS/SARIMAX.md for all options and details.
       Smoke output goes to OUTPUT/sarimax/revised/smoke/ unless --output is given.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('pedestrians', 'cyclists', 'motorists')
EXOG = ['holiday_week', 'covid_period']
# Deliberately small grid; declaration order breaks exact MAE ties.
CANDIDATES = [((1, 0, 0), (0, 0, 0, 52)),
              ((1, 0, 0), (1, 0, 0, 52)),
              ((0, 1, 1), (0, 1, 0, 52))]
REVISED_CONFIGS = {
    'pedestrians': {'name': 'seasonal_smoothing_ar', 'order': (1, 0, 1),
                    'seasonal_order': (0, 1, 1, 52), 'trend': 'n'},
    'cyclists': {'name': 'seasonal_smoothing', 'order': (0, 0, 1),
                'seasonal_order': (0, 1, 1, 52), 'trend': 'n'},
    'motorists': {'name': 'original', 'order': (0, 1, 1),
                 'seasonal_order': (0, 1, 0, 52), 'trend': 'c'},
}


def model_config(candidate):
    """Accept legacy order pairs as well as explicit per-model trend choices."""
    if isinstance(candidate, dict):
        return {k: candidate[k] for k in ('order', 'seasonal_order', 'trend')}
    return {'order': candidate[0], 'seasonal_order': candidate[1], 'trend': 'c'}


def profile_configs(profile, group):
    if profile == 'original':
        return [dict(model_config(c), name=f'original_grid_{i}') for i, c in enumerate(CANDIDATES)]
    if profile == 'revised':
        return [REVISED_CONFIGS[group].copy()]
    raise ValueError(f'Unknown profile: {profile}')


def load_data(path):
    df = pd.read_csv(path, index_col='week_ending', parse_dates=True)
    expected = pd.date_range('2016-03-13', '2025-12-28', freq='W-SUN', name='week_ending')
    if not df.index.equals(expected):
        raise ValueError('Expected unique, ordered, complete W-SUN study index')
    df = df.asfreq('W-SUN')
    values = df[list(GROUPS) + EXOG].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy()).all() or (values[list(GROUPS)] < 0).any().any():
        raise ValueError('Missing/nonfinite values or negative injury counts')
    if not values[EXOG].isin([0, 1]).all().all():
        raise ValueError('Indicators must be binary')
    if any(values[c].nunique() < 2 for c in EXOG) or np.linalg.matrix_rank(values[EXOG]) != 2:
        raise ValueError('Constant or redundant exogenous columns')
    return values


def splits(n):
    """Full 12-week folds only, strictly before the final 12 observations."""
    return [(slice(0, end), slice(end, end + 12)) for end in range(280, n - 23, 12)]


def fit_forecast(train, future_exog, group, candidate, maxiter):
    if list(future_exog.columns) != EXOG or len(future_exog) != 12:
        raise ValueError('Forecast exog must have 12 rows and declared column order')
    expected = pd.date_range(train.index[-1] + pd.Timedelta(weeks=1), periods=12, freq='W-SUN')
    if not future_exog.index.equals(expected) or not np.isfinite(future_exog.to_numpy()).all():
        raise ValueError('Forecast exog dates/values invalid')
    started = time.perf_counter()
    diagnostic = {'converged': False, 'error': None, 'warnings': []}
    forecast = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model = SARIMAX(train[group], exog=train[EXOG], **model_config(candidate),
                            enforce_stationarity=True, enforce_invertibility=True)
            result = model.fit(disp=False, maxiter=maxiter)
            diagnostic['converged'] = bool(result.mle_retvals.get('converged', False))
            diagnostic['iterations'] = int(result.mle_retvals.get('iterations', -1))
            resid = np.asarray(result.resid)[max(52, result.loglikelihood_burn):]
            diagnostic['residual_mean'] = float(resid.mean())
            diagnostic['residual_std'] = float(resid.std())
            diagnostic['ljung_box'] = np.asarray(result.test_serial_correlation('ljungbox', lags=[12, 52])).tolist()
            prediction = result.get_forecast(steps=12, exog=future_exog)
            bounds = np.asarray(prediction.conf_int(alpha=0.20))
            forecast = pd.DataFrame({'predicted': np.asarray(prediction.predicted_mean),
                                     'lower_80': bounds[:, 0], 'upper_80': bounds[:, 1]}, index=future_exog.index)
            if not np.isfinite(forecast.to_numpy()).all() or (forecast.lower_80 > forecast.upper_80).any():
                raise ValueError('Invalid forecast values or intervals')
        except Exception as exc:
            diagnostic['error'] = f'{type(exc).__name__}: {exc}'
            forecast = None
        diagnostic['warnings'] = [f'{type(w.message).__name__}: {w.message}' for w in caught]
    diagnostic['elapsed_seconds'] = time.perf_counter() - started
    return forecast, diagnostic


def select(scores, fold_count):
    eligible = [s for s in scores if s['valid_folds'] == fold_count]
    if not eligible:
        raise RuntimeError('No candidate converged successfully on every fold; see diagnostics.json')
    return min(eligible, key=lambda s: (s['mean_mae'], s['candidate']))


def validate_output(output, test_index):
    if len(output) != 36 or set(output['group']) != set(GROUPS):
        raise ValueError('Expected three groups and 36 forecasts')
    for _, rows in output.groupby('group'):
        if not pd.DatetimeIndex(rows.week_ending).equals(test_index):
            raise ValueError('Forecast dates misaligned')
    if not np.isfinite(output[['actual', 'predicted', 'lower_80', 'upper_80', 'seasonal_naive']].to_numpy()).all():
        raise ValueError('Nonfinite output')
    if (output.lower_80 > output.upper_80).any():
        raise ValueError('Reversed interval bounds')


def run(args):
    np.random.seed(4002)
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    df = load_data(args.input)
    folds = splits(len(df))
    configs = {group: profile_configs(args.profile, group) for group in GROUPS}
    if args.profile == 'revised' and args.output.resolve() == (ROOT / 'OUTPUT/sarimax/full').resolve():
        raise ValueError('Revised runs must not overwrite the original study directory')
    if (args.output / 'run.json').exists() or (args.output / 'forecasts.csv').exists():
        raise ValueError('Output already contains a run; use a fresh --output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'status': 'running', 'input_sha256': digest, 'mode': args.mode, 'maxiter': args.maxiter,
                'profile': args.profile, 'jobs': args.jobs,
                'seed': 4002, 'model_configs': configs, 'exog': EXOG,
                'enforce_stationarity': True, 'enforce_invertibility': True,
                'selection': ('all folds converged; minimum mean MAE; ties by candidate index'
                              if args.profile == 'original' else
                              'fixed group configurations adopted from pretest investigation; all 18 folds must converge'),
                'interpretation': ('original study' if args.profile == 'original' else
                                   'exploratory follow-up; 2025 holdout was previously examined, not independent confirmation'),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']},
                'test_dates': [str(d.date()) for d in df.index[-12:]],
                'folds': [{'train_start': str(df.index[a.start].date()), 'train_end': str(df.index[a.stop-1].date()),
                           'validation_start': str(df.index[b.start].date()), 'validation_end': str(df.index[b.stop-1].date())} for a,b in folds]}
    diagnostics, scores, summaries, forecasts, best = [], [], [], [], {}
    validation_forecasts = []

    def save():
        (args.output / 'run.json').write_text(json.dumps(metadata, indent=2))
        (args.output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        frame = pd.DataFrame(scores)
        if len(frame):
            frame = frame.sort_values(['group', 'candidate', 'fold'])
        frame.to_csv(args.output / 'fold_scores.csv', index=False)
        pd.DataFrame(summaries).to_csv(args.output / 'candidate_scores.csv', index=False)
        if validation_forecasts:
            pd.concat(validation_forecasts).sort_values(['group', 'candidate', 'fold', 'week_ending']).to_csv(
                args.output / 'validation_forecasts.csv', index=False)

    def consume(task, result):
        group, c, fold, a, b = task
        pred, diag = result
        name = configs[group][c]['name']
        diagnostics.append(dict(diag, group=group, candidate=c, candidate_name=name, fold=fold,
                                train_end=str(df.index[a.stop-1].date()),
                                validation_start=str(df.index[b.start].date()),
                                validation_end=str(df.index[b.stop-1].date())))
        ok = pred is not None and diag['converged'] and diag['error'] is None
        score = {'group': group, 'candidate': c, 'candidate_name': name,
                 'fold': fold, 'mae': None, 'valid': ok,
                 'baseline_mae': float(mean_absolute_error(df.iloc[b][group], df[group].shift(52).iloc[b]))}
        if pred is not None:
            pred = pred.copy()
            pred['actual'] = df.iloc[b][group]
            pred['seasonal_naive'] = df[group].shift(52).iloc[b]
            score.update(mae=float(mean_absolute_error(pred.actual, pred.predicted)),
                         bias=float((pred.predicted-pred.actual).mean()),
                         coverage=float(((pred.actual >= pred.lower_80) & (pred.actual <= pred.upper_80)).mean()),
                         interval_width=float((pred.upper_80-pred.lower_80).mean()))
            pred['group'], pred['candidate'], pred['candidate_name'], pred['fold'] = group, c, name, fold
            validation_forecasts.append(pred.reset_index())
        scores.append(score)
        print(f'{group} {name} fold={fold} MAE={score["mae"]} converged={diag["converged"]}', flush=True)
        save()
              
    try:
        if args.mode == 'smoke':
            a, b = folds[0]
            group = 'cyclists' if args.profile == 'revised' else 'pedestrians'
            c = 0
            pred, diag = fit_forecast(df.iloc[a], df.iloc[b][EXOG], group, configs[group][c], args.maxiter)
            diagnostics.append(dict(diag, group=group, candidate=c, candidate_name=configs[group][c]['name'], fold=0))
            print(json.dumps(diagnostics[-1]), flush=True)
            if pred is None or not diag['converged']:
                raise RuntimeError('Smoke test failed or did not converge; see diagnostics')
            pred.to_csv(args.output / f'smoke_{group}_candidate_{c}.csv', index_label='week_ending')
            metadata['status'] = 'complete'
            return
        tasks = [(group, c, fold, a, b) for group in GROUPS
                 for c in range(len(configs[group])) for fold, (a, b) in enumerate(folds)]
        save()
        if args.jobs == 1:
            for task in tasks:
                group, c, _, a, b = task
                consume(task, fit_forecast(df.iloc[a], df.iloc[b][EXOG], group, configs[group][c], args.maxiter))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                futures = {}
                for task in tasks:
                    group, c, _, a, b = task
                    future = pool.submit(fit_forecast, df.iloc[a], df.iloc[b][EXOG],
                                         group, configs[group][c], args.maxiter)
                    futures[future] = task
                for future in as_completed(futures):
                    consume(futures[future], future.result())
        for group in GROUPS:
            group_scores = []
            for c, config in enumerate(configs[group]):
                rows = [s for s in scores if s['group'] == group and s['candidate'] == c]
                valid = [s for s in rows if s['valid']]
                complete = len(valid) == len(folds)
                summary = {'group': group, 'candidate': c, 'candidate_name': config['name'],
                           'valid_folds': len(valid),
                           'mean_mae': float(np.mean([s['mae'] for s in valid])) if complete else None,
                           'baseline_mae': float(np.mean([s['baseline_mae'] for s in rows])),
                           'coverage': float(np.mean([s['coverage'] for s in valid])) if complete else None,
                           'mean_interval_width': float(np.mean([s['interval_width'] for s in valid])) if complete else None}
                group_scores.append(summary)
                summaries.append(summary)
            best[group] = select(group_scores, len(folds))
        # This file is written before any final test targets are evaluated.
        (args.output / 'best_orders.json').write_text(json.dumps(
            {g: dict(s, **model_config(configs[g][s['candidate']])) for g, s in best.items()}, indent=2))
        if args.mode == 'validate':
            metadata['status'] = 'complete'
            return
        for group in GROUPS:
            config = configs[group][best[group]['candidate']]
            pred, diag = fit_forecast(df.iloc[:-12], df.iloc[-12:][EXOG], group, config, args.maxiter)
            diagnostics.append(dict(diag, group=group, candidate_name=config['name'], stage='final'))
            if pred is None or not diag['converged']:
                raise RuntimeError(f'Final fit failed/nonconverged for {group}; see diagnostics')
            pred['actual'] = df.iloc[-12:][group]
            pred['seasonal_naive'] = df[group].shift(52).iloc[-12:]
            pred['group'] = group
            forecasts.append(pred)
        output = pd.concat(forecasts).reset_index()
        validate_output(output, df.index[-12:])
        output.to_csv(args.output / 'forecasts.csv', index=False)
        metadata['negative_point_forecasts'] = int((output.predicted < 0).sum())
        metadata['negative_lower_bounds'] = int((output.lower_80 < 0).sum())
        metadata['status'] = 'complete'
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        changed = hashlib.sha256(args.input.read_bytes()).hexdigest() != digest
        if changed:
            metadata['status'] = 'failed'
            metadata['error'] = 'Processed input changed during execution'
        save()
        if changed:
            raise RuntimeError('Processed input changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'DATA/weekly_injuries_processed.csv')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--mode', choices=['smoke', 'validate', 'full'], default='smoke')
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--profile', choices=['original', 'revised'], default='revised')
    parser.add_argument('--jobs', type=int, choices=range(1, 4), default=1,
                        help='Parallel CV fits (1–3); final fits remain serial')
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error('--maxiter must be positive')
    base = ROOT / 'OUTPUT/sarimax'
    if args.profile == 'revised':
        base = base / 'revised'
    args.output = args.output or base / args.mode
    run(args)
