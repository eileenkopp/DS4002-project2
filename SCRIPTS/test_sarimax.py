"""Run: python -m unittest discover -s SCRIPTS -p 'test_*.py'."""
import hashlib
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

import sarimax as pipeline


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


if __name__ == '__main__':
    unittest.main()
