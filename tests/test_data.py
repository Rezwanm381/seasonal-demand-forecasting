import unittest

import pandas as pd

from src.data_prep import audit_series, validate_audit


def valid_frame(periods=range(1, 9)):
    periods = list(periods)
    return pd.DataFrame(
        {
            "period_index": periods,
            "period_label": [f"P{value}" for value in periods],
            "year": [(value - 1) // 4 + 1 for value in periods],
            "quarter": [(value - 1) % 4 + 1 for value in periods],
            "demand": [float(value * 2) for value in periods],
        }
    )


class DataAuditTests(unittest.TestCase):
    def test_missing_period_detection(self):
        audit = audit_series(valid_frame([1, 2, 4, 5]))
        self.assertEqual(audit.missing_periods, [3])
        self.assertTrue(audit.irregular_spacing)

    def test_chronological_order_detection(self):
        audit = audit_series(valid_frame([1, 3, 2, 4]))
        self.assertFalse(audit.chronological_order)

    def test_clean_generic_quarterly_sequence(self):
        audit = audit_series(valid_frame())
        self.assertTrue(audit.chronological_order)
        self.assertEqual(audit.duplicate_periods, [])
        self.assertEqual(audit.missing_periods, [])
        self.assertTrue(audit.quarter_mapping_consistent)

    def test_duplicate_period_detection(self):
        audit = audit_series(valid_frame([1, 2, 2, 4]))
        self.assertEqual(audit.duplicate_periods, [2])
        with self.assertRaises(ValueError):
            validate_audit(audit)

    def test_non_numeric_target_is_rejected_by_audit(self):
        frame = valid_frame()
        frame["demand"] = frame["demand"].astype(object)
        frame.loc[2, "demand"] = "not-a-number"
        audit = audit_series(frame)
        self.assertEqual(audit.missing_target_values, 1)
        with self.assertRaises(ValueError):
            validate_audit(audit)

    def test_missing_period_value_is_reported_without_conversion_crash(self):
        frame = valid_frame()
        frame.loc[2, "period_index"] = None
        audit = audit_series(frame)
        self.assertFalse(audit.chronological_order)
        self.assertTrue(audit.irregular_spacing)
        with self.assertRaises(ValueError):
            validate_audit(audit)


if __name__ == "__main__":
    unittest.main()
