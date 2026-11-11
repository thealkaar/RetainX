import unittest
from datetime import datetime, timedelta

from train_models import INACTIVITY_DAYS, make_snapshots


class NinetyDayChurnLabelTests(unittest.TestCase):
    def test_snapshot_only_uses_past_orders_and_next_90_day_outcome(self):
        base = datetime(2020, 1, 1)
        def order(day):
            return {"date": base + timedelta(days=day), "order_value": 100.0,
                    "freight_value": 10.0, "total_items": 1,
                    "payment_installments": 1, "payment_type": "credit_card",
                    "review_score": 5, "delivery_duration_days": 5,
                    "delivery_delay_days": 0}
        rows = {"repeat": [order(0), order(60), order(140)],
                "inactive": [order(0)]}
        features, labels, groups = make_snapshots(rows, base + timedelta(days=250), step_days=30)
        self.assertTrue(features)
        self.assertEqual(INACTIVITY_DAYS, 90)
        for feature in features:
            self.assertIn("days_since_last_order", feature)
            self.assertNotIn("future_order_count", feature)
        self.assertEqual(len(features), len(labels))
        self.assertEqual(len(features), len(groups))
        self.assertIn(0, labels)
        self.assertIn(1, labels)


if __name__ == "__main__":
    unittest.main()
