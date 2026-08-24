import unittest

import numpy as np

from src.evaluation import assert_no_temporal_leakage, expanding_window_evaluate, model_comparison
from src.models import seasonal_mean, seasonal_naive


# Synthetic fixture used only to test seasonal indexing and leakage controls.
# It intentionally does not reproduce the rights-restricted course observations.
SERIES = np.array([10, 20, 50, 30, 12, 22, 52, 32, 11, 21, 51, 31], dtype=float)


class EvaluationTests(unittest.TestCase):
    def test_seasonal_naive_forecast(self):
        result = seasonal_naive([10, 20, 30, 40, 11, 21, 31, 41], 4, 4)
        np.testing.assert_allclose(result, [11, 21, 31, 41])

    def test_seasonal_naive_partial_cycle_origins(self):
        expected = {
            9: [22, 52, 32],
            10: [52, 32],
            11: [32],
        }
        for origin, forecast_values in expected.items():
            horizon = len(SERIES) - origin
            with self.subTest(origin=origin):
                np.testing.assert_allclose(
                    seasonal_naive(SERIES[:origin], horizon, 4), forecast_values
                )

    def test_seasonal_mean_uses_available_quarter_values(self):
        result = seasonal_mean(SERIES[:8], 4, 4)
        np.testing.assert_allclose(result, [11.0, 21.0, 51.0, 31.0])

    def test_expanding_window_origin_and_horizon_counts(self):
        result = expanding_window_evaluate(
            SERIES, models=["SEASONAL_NAIVE"], initial_train_size=8, max_horizon=4
        )
        self.assertEqual(sorted(result["origin_period"].unique().tolist()), [8, 9, 10, 11])
        self.assertEqual(len(result), 10)
        self.assertEqual(result.groupby("horizon").size().to_dict(), {1: 4, 2: 3, 3: 2, 4: 1})

    def test_every_stored_split_passes_strict_leakage_rule(self):
        result = expanding_window_evaluate(SERIES, initial_train_size=8, max_horizon=4)
        self.assertTrue((result["train_end_period"] < result["target_period"]).all())
        self.assertTrue(result["leakage_check"].all())

    def test_future_value_cannot_change_any_backtest_prediction(self):
        changed = SERIES.copy()
        changed[-1] = 9999
        original_results = expanding_window_evaluate(SERIES, initial_train_size=8, max_horizon=4)
        changed_results = expanding_window_evaluate(changed, initial_train_size=8, max_horizon=4)
        np.testing.assert_allclose(original_results["prediction"], changed_results["prediction"])

    def test_leakage_assertion_stops_overlap(self):
        with self.assertRaises(AssertionError):
            assert_no_temporal_leakage([1, 2, 3], [3, 4])

    def test_one_step_robustness_retains_seasonal_mean(self):
        predictions = expanding_window_evaluate(
            SERIES, initial_train_size=8, max_horizon=1
        )
        comparison = model_comparison(predictions)
        self.assertEqual(comparison.iloc[0]["Model"], "SEASONAL_MEAN")


if __name__ == "__main__":
    unittest.main()
