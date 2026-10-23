"""Weekly injury SARIMAX; run from any directory. See SARIMAX.md."""
import argparse
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
            model = SARIMAX(train[group], exog=train[EXOG], order=candidate[0],
                            seasonal_order=candidate[1], trend='c',
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
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'status': 'running', 'input_sha256': digest, 'mode': args.mode, 'maxiter': args.maxiter,
                'seed': 4002, 'candidates': CANDIDATES, 'exog': EXOG,
                'trend': 'c', 'enforce_stationarity': True, 'enforce_invertibility': True,
                'selection': 'all folds converged; minimum mean MAE; ties by candidate index',
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn']},
                'test_dates': [str(d.date()) for d in df.index[-12:]],
                'folds': [{'train_start': str(df.index[a.start].date()), 'train_end': str(df.index[a.stop-1].date()),
                           'validation_start': str(df.index[b.start].date()), 'validation_end': str(df.index[b.stop-1].date())} for a,b in folds]}
    diagnostics, scores, summaries, forecasts, best = [], [], [], [], {}

    def save():
        (args.output / 'run.json').write_text(json.dumps(metadata, indent=2))
        (args.output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        pd.DataFrame(scores).to_csv(args.output / 'fold_scores.csv', index=False)
        pd.DataFrame(summaries).to_csv(args.output / 'candidate_scores.csv', index=False)

    try:
        if args.mode in ('smoke', 'benchmark'):
            a, b = folds[0]
            indices = [0] if args.mode == 'smoke' else list(range(len(CANDIDATES)))
            for c in indices:
                pred, diag = fit_forecast(df.iloc[a], df.iloc[b][EXOG], GROUPS[0], CANDIDATES[c], args.maxiter)
                diagnostics.append(dict(diag, group=GROUPS[0], candidate=c, fold=0))
                print(json.dumps(diagnostics[-1]), flush=True)
                if pred is None:
                    raise RuntimeError('Smoke/benchmark failed; see diagnostics')
                pred.to_csv(args.output / f'benchmark_candidate_{c}.csv', index_label='week_ending')
            metadata['status'] = 'complete'
            return
        for group in GROUPS:
            group_scores = []
            for c, candidate in enumerate(CANDIDATES):
                valid = []
                for fold, (a, b) in enumerate(folds):
                    pred, diag = fit_forecast(df.iloc[a], df.iloc[b][EXOG], group, candidate, args.maxiter)
                    diagnostics.append(dict(diag, group=group, candidate=c, fold=fold))
                    mae = float(mean_absolute_error(df.iloc[b][group], pred.predicted)) if pred is not None else None
                    ok = pred is not None and diag['converged'] and diag['error'] is None
                    if ok:
                        valid.append(mae)
                    scores.append({'group': group, 'candidate': c, 'fold': fold, 'mae': mae, 'valid': ok})
                    print(f'{group} candidate={c} fold={fold} MAE={mae} converged={diag["converged"]}', flush=True)
                    save()
                summary = {'group': group, 'candidate': c, 'valid_folds': len(valid),
                           'mean_mae': float(np.mean(valid)) if len(valid) == len(folds) else None}
                group_scores.append(summary)
                summaries.append(summary)
            best[group] = select(group_scores, len(folds))
        (args.output / 'best_orders.json').write_text(json.dumps({g: dict(s, order=CANDIDATES[s['candidate']][0], seasonal_order=CANDIDATES[s['candidate']][1]) for g,s in best.items()}, indent=2))
        for group in GROUPS:
            pred, diag = fit_forecast(df.iloc[:-12], df.iloc[-12:][EXOG], group, CANDIDATES[best[group]['candidate']], args.maxiter)
            diagnostics.append(dict(diag, group=group, stage='final'))
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
        save()
        if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Processed input changed during execution')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'DATA/weekly_injuries_processed.csv')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--mode', choices=['smoke', 'benchmark', 'full'], default='smoke')
    parser.add_argument('--maxiter', type=int, default=200)
    args = parser.parse_args()
    if args.maxiter < 1:
        parser.error('--maxiter must be positive')
    args.output = args.output or ROOT / 'OUTPUT/sarimax' / args.mode
    run(args)
