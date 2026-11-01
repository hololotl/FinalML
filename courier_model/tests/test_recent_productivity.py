import unittest

import pandas as pd

from courier_model import recent_productivity as rp


class RecentProductivityTests(unittest.TestCase):
    def test_slots_follow_forecast_orders_over_recent_productivity(self):
        dates = pd.date_range("2026-09-01", periods=4)  # Tue..Fri, all weekdays
        hour_segments = pd.DataFrame(
            {"location_id": ["1", "1"], "hour": [12, 13], "time_segment": ["lunch", "lunch"]}
        )
        orders = pd.DataFrame(
            [
                {"location_id": "1", "date": d, "hour": h, "vehicle_type": "auto", "orders": 4}
                for d in dates for h in (12, 13)
            ]
        )
        couriers = pd.DataFrame(
            [
                {"location_id": "1", "date": d, "hour": h, "vehicle_type": "auto", "couriers": 2}
                for d in dates for h in (12, 13)
            ]
        )
        old_min = rp.MIN_COURIER_HOURS
        rp.MIN_COURIER_HOURS = 1
        try:
            table = rp.build_productivity_table(orders, couriers, hour_segments, dates)
            # 4 orders per courier-hour pair of 2 couriers -> productivity 2 orders/courier/hour
            self.assertAlmostEqual(float(table["productivity"].iloc[0]), 2.0, places=3)
            hourly = pd.DataFrame(
                [
                    {"location_id": "1", "date": "2026-09-08", "hour": h,
                     "vehicle_type": "auto", "time_segment": "lunch",
                     "slots_needed": 1, "predicted_orders": 8.0}
                    for h in (12, 13)
                ]
            )
            result, _ = rp.apply_recent_productivity(hourly, table)
        finally:
            rp.MIN_COURIER_HOURS = old_min
        # 8 orders / 2 per courier = 4 couriers
        self.assertEqual(result["slots_needed"].tolist(), [4, 4])
        self.assertEqual(result["slots_needed_model"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
