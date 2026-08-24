import unittest

from src.metrics import mae, mase, rmse, seasonal_mase_scale, smape


class MetricTests(unittest.TestCase):
    def test_mae_and_rmse(self):
        actual = [1, 2, 3]
        predicted = [1, 4, 2]
        self.assertAlmostEqual(mae(actual, predicted), 1.0)
        self.assertAlmostEqual(rmse(actual, predicted), (5 / 3) ** 0.5)

    def test_seasonal_mase_scale_uses_training_window(self):
        train = [10, 20, 30, 40, 12, 22, 32, 42]
        self.assertAlmostEqual(seasonal_mase_scale(train, 4), 2.0)
        self.assertAlmostEqual(mase([14], [12], train, 4), 1.0)

    def test_smape(self):
        self.assertAlmostEqual(smape([100], [110]), 200 * 10 / 210)

    def test_seasonal_mase_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            seasonal_mase_scale([10, 20, 30, 40], 4)

    def test_seasonal_mase_rejects_zero_scale(self):
        with self.assertRaises(ValueError):
            seasonal_mase_scale([10, 20, 30, 40, 10, 20, 30, 40], 4)


if __name__ == "__main__":
    unittest.main()
