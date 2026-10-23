"""Run: python -m unittest discover -s SCRIPTS -p 'test_*.py'."""
import hashlib
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

import sarimax as pipeline
from check_intervals import history_check


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.path = pipeline.ROOT / 'DATA/weekly_injuries_processed.csv'
        self.df = pipeline.load_data(self.path)

    def test_schema_and_split_boundaries(self):
        self.assertEqual(len(pipeline.GROUPS), 3)
        self.assertEqual(len(self.df), 512)
        self.assertEqual(str(self.df.index[-12].date()), '2025-10-12')
        self.assertEqual(str(self.df.index[-1].date()), '2025-12-28')
        folds = pipeline.splits(len(self.df))
        self.assertEqual(len(folds), 18)
        for a, b in folds:
            self.assertEqual(b.stop - b.start, 12)
            self.assertEqual(a.stop, b.start)
            self.assertLessEqual(b.stop, len(self.df) - 12)
        self.assertEqual(folds[0][0].stop, 280)
        self.assertEqual(folds[-1][1].stop, 496)

    def test_bad_inputs_rejected(self):
        a, b = pipeline.splits(len(self.df))[0]
        with self.assertRaises(ValueError):
            pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG[::-1]],
                                  'pedestrians', pipeline.CANDIDATES[0], 1)
        with self.assertRaises(ValueError):
            pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG].iloc[:-1],
                                  'pedestrians', pipeline.CANDIDATES[0], 1)

    def test_selection_requires_all_folds_and_breaks_ties(self):
        scores = [{'candidate': 2, 'valid_folds': 18, 'mean_mae': 10},
                  {'candidate': 1, 'valid_folds': 18, 'mean_mae': 10},
                  {'candidate': 0, 'valid_folds': 17, 'mean_mae': 1}]
        self.assertEqual(pipeline.select(scores, 18)['candidate'], 1)
        with self.assertRaises(RuntimeError):
            pipeline.select(scores, 19)

    def test_one_forecast_call_without_validation_targets(self):
        a, b = pipeline.splits(len(self.df))[0]
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch.object(pipeline, 'SARIMAX') as model:
            result = model.return_value.fit.return_value
            result.mle_retvals = {'converged': True, 'iterations': 3}
            result.resid = np.zeros(280)
            result.loglikelihood_burn = 0
            result.test_serial_correlation.return_value = np.zeros((1, 2, 2))
            result.get_forecast.return_value.predicted_mean = np.ones(12)
            result.get_forecast.return_value.conf_int.return_value = np.tile([0., 2.], (12, 1))
            pred, diag = pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG],
                                              'pedestrians', pipeline.CANDIDATES[0], 10)
            pd.testing.assert_series_equal(model.call_args.args[0], self.df.iloc[a].pedestrians)
            result.get_forecast.assert_called_once()
            self.assertEqual(result.get_forecast.call_args.kwargs['steps'], 12)
            pd.testing.assert_frame_equal(result.get_forecast.call_args.kwargs['exog'], self.df.iloc[b][pipeline.EXOG])
            result.get_forecast.return_value.conf_int.assert_called_once_with(alpha=0.20)
            self.assertEqual(len(pred), 12)
            self.assertTrue(diag['converged'])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)

    def test_failures_and_nonconvergence_recorded(self):
        a, b = pipeline.splits(len(self.df))[0]
        with patch.object(pipeline, 'SARIMAX', side_effect=RuntimeError('fit failed')):
            pred, diag = pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG],
                                              'pedestrians', pipeline.CANDIDATES[0], 1)
        self.assertIsNone(pred)
        self.assertFalse(diag['converged'])
        self.assertIn('fit failed', diag['error'])
        with patch.object(pipeline, 'SARIMAX') as model:
            def fail(**kwargs):
                warnings.warn('optimizer did not converge', RuntimeWarning)
                raise RuntimeError('optimizer failure')
            model.return_value.fit.side_effect = fail
            _, diag = pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG],
                                            'pedestrians', pipeline.CANDIDATES[0], 1)
        self.assertFalse(diag['converged'])
        self.assertIn('optimizer did not converge', diag['warnings'][0])

    def test_final_output_alignment(self):
        rows = pd.concat([pd.DataFrame({'week_ending': self.df.index[-12:],
                                       'group': g, 'actual': 1., 'predicted': 1.,
                                       'lower_80': 0., 'upper_80': 2., 'seasonal_naive': 1.})
                          for g in pipeline.GROUPS], ignore_index=True)
        pipeline.validate_output(rows, self.df.index[-12:])
        with self.assertRaises(ValueError):
            pipeline.validate_output(rows.iloc[:-1], self.df.index[-12:])
        rows.loc[0, 'lower_80'] = 3.
        with self.assertRaises(ValueError):
            pipeline.validate_output(rows, self.df.index[-12:])

    def test_profile_and_trend_are_explicit(self):
        cyclist = pipeline.profile_configs('revised', 'cyclists')[0]
        self.assertEqual(pipeline.model_config(cyclist),
                         {'order': (0, 0, 1), 'seasonal_order': (0, 1, 1, 52), 'trend': 'n'})
        self.assertEqual(pipeline.profile_configs('revised', 'motorists')[0]['trend'], 'c')
        self.assertEqual(pipeline.model_config(pipeline.profile_configs('revised', 'pedestrians')[0]),
                         {'order': (1, 0, 1), 'seasonal_order': (0, 1, 1, 52), 'trend': 'n'})
        self.assertEqual(len(pipeline.profile_configs('original', 'cyclists')), 3)
        self.assertEqual(pipeline.model_config(pipeline.CANDIDATES[2])['trend'], 'c')
        a, b = pipeline.splits(len(self.df))[0]
        with patch.object(pipeline, 'SARIMAX', side_effect=RuntimeError('stop after construction')) as model:
            pipeline.fit_forecast(self.df.iloc[a], self.df.iloc[b][pipeline.EXOG], 'cyclists', cyclist, 1)
        self.assertEqual(model.call_args.kwargs['trend'], 'n')
        self.assertEqual(model.call_args.kwargs['seasonal_order'], (0, 1, 1, 52))

    def fake_forecast(self, train, exog, group, config, maxiter):
        self.calls.append((group, len(train), train.index[-1], exog.index.copy(), config))
        pred = pd.DataFrame({'predicted': np.full(12, 100.), 'lower_80': np.zeros(12),
                             'upper_80': np.full(12, 200.)}, index=exog.index)
        return pred, {'converged': True, 'error': None, 'warnings': [], 'elapsed_seconds': 0.}

    def args_for(self, output):
        return argparse.Namespace(input=self.path, output=Path(output), profile='revised',
                                  mode='full', maxiter=1, jobs=1)

    def test_full_revised_run_separates_cv_from_final_and_preserves_input(self):
        self.calls = []
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as output:
            args = self.args_for(output)
            with patch.object(pipeline, 'fit_forecast', side_effect=self.fake_forecast), redirect_stdout(io.StringIO()):
                pipeline.run(args)
            self.assertEqual(len(self.calls), 57)  # 54 CV fits, then 3 final fits.
            self.assertTrue(all(n < 500 and dates[-1] < self.df.index[-12]
                                for _, n, _, dates, _ in self.calls[:54]))
            self.assertTrue(all(n == 500 for _, n, _, _, _ in self.calls[54:]))
            for _, _, end, dates, _ in self.calls:
                self.assertEqual(len(dates), 12)
                self.assertEqual(end + pd.Timedelta(weeks=1), dates[0])
            forecasts = pd.read_csv(Path(output)/'forecasts.csv', parse_dates=['week_ending'])
            pipeline.validate_output(forecasts, self.df.index[-12:])
            validation = pd.read_csv(Path(output)/'validation_forecasts.csv', parse_dates=['week_ending'])
            self.assertEqual(len(validation), 3*18*12)
            self.assertTrue((validation.week_ending < self.df.index[-12]).all())
            manifest = json.loads((Path(output)/'run.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['profile'], 'revised')
            self.assertIn('not independent', manifest['interpretation'])
            export_hash = hashlib.sha256((Path(output)/'forecasts.csv').read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                pipeline.run(args)
            self.assertEqual(hashlib.sha256((Path(output)/'forecasts.csv').read_bytes()).hexdigest(), export_hash)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)

    def test_failed_cv_prevents_final_forecast_export(self):
        self.calls = []
        def failing_fit(*args):
            pred, diag = self.fake_forecast(*args)
            if args[2] == 'cyclists' and len(args[0]) == 280:
                diag['converged'] = False
            return pred, diag
        with tempfile.TemporaryDirectory() as output:
            with patch.object(pipeline, 'fit_forecast', side_effect=failing_fit), redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    pipeline.run(self.args_for(output))
            self.assertFalse((Path(output)/'forecasts.csv').exists())
            self.assertTrue(all(n < 500 for _, n, _, _, _ in self.calls))
            self.assertEqual(json.loads((Path(output)/'run.json').read_text())['status'], 'failed')

    def test_validation_only_mode_never_fits_or_exports_test_forecasts(self):
        self.calls = []
        with tempfile.TemporaryDirectory() as output:
            args = self.args_for(output)
            args.mode = 'validate'
            with patch.object(pipeline, 'fit_forecast', side_effect=self.fake_forecast), redirect_stdout(io.StringIO()):
                pipeline.run(args)
            self.assertEqual(len(self.calls), 54)
            self.assertTrue(all(n < 500 for _, n, _, _, _ in self.calls))
            self.assertFalse((Path(output)/'forecasts.csv').exists())
            self.assertTrue((Path(output)/'best_orders.json').exists())

    def test_interval_history_uses_only_previous_folds_and_never_shrinks(self):
        rows = pd.DataFrame({'week_ending': pd.date_range('2021-07-25', periods=216, freq='W-SUN'),
                             'fold': np.repeat(np.arange(18), 12), 'predicted': 100.,
                             'lower_80': 90., 'upper_80': 110., 'actual': 100.})
        rows.loc[:71, 'actual'] = 112.
        initial = history_check(rows)
        self.assertEqual(initial.iloc[0].history_observations, 72)
        self.assertAlmostEqual(initial.iloc[0].widening_factor, 1.2)
        self.assertTrue((initial.widening_factor >= 1).all())
        rows.loc[rows.fold == 17, 'actual'] = 1000.
        altered = history_check(rows)
        np.testing.assert_array_equal(initial.widening_factor, altered.widening_factor)
        self.assertNotEqual(initial.iloc[-1].adjusted_coverage, altered.iloc[-1].adjusted_coverage)
        rows.loc[215, 'week_ending'] = pd.Timestamp('2025-10-12')
        with self.assertRaises(ValueError):
            history_check(rows)

    def test_changed_input_cannot_leave_a_successful_run_status(self):
        self.calls = []
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'input.csv'
            source.write_bytes(self.path.read_bytes())
            args = self.args_for(Path(folder)/'outputs')
            args.input = source
            args.mode = 'validate'
            def changing_fit(*values):
                result = self.fake_forecast(*values)
                source.write_text('changed during execution')
                return result
            with patch.object(pipeline, 'fit_forecast', side_effect=changing_fit), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Processed input changed'):
                    pipeline.run(args)
            self.assertEqual(json.loads((args.output/'run.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
