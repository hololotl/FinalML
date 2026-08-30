import unittest
from unittest.mock import patch

import pandas as pd

from courier_model import delivery_duration_report
from courier_model import main as courier_main


class DeliveryCapacityTests(unittest.TestCase):
    def test_cycle_time_wins_before_single_safety_rounding(self):
        with patch.object(courier_main, "SAFETY_BUFFER", 1.15):
            (
                history_raw,
                cycle_raw,
                selected_raw,
                selected_source,
                auto_slots,
                bike_slots,
            ) = courier_main.select_slot_demand(
                auto_orders=8.0,
                bike_orders=0.0,
                slot_capacity={
                    courier_main.AUTO: 4.0,
                    courier_main.BIKE: 2.0,
                },
                cycle_minutes={
                    courier_main.AUTO: 30.0,
                    courier_main.BIKE: 30.0,
                },
                window_hours=2.0,
            )

        self.assertEqual(history_raw[courier_main.AUTO], 2.0)
        self.assertEqual(cycle_raw[courier_main.AUTO], 4.0)
        self.assertEqual(selected_raw[courier_main.AUTO], 4.0)
        self.assertEqual(selected_source[courier_main.AUTO], "cycle_time")
        self.assertEqual(auto_slots, 5)
        self.assertEqual(bike_slots, 0)

    def test_history_wins_when_it_is_more_conservative(self):
        result = courier_main.select_slot_demand(
            auto_orders=6.0,
            bike_orders=0.0,
            slot_capacity={
                courier_main.AUTO: 1.5,
                courier_main.BIKE: 1.0,
            },
            cycle_minutes={
                courier_main.AUTO: 10.0,
                courier_main.BIKE: 10.0,
            },
            window_hours=3.0,
        )
        self.assertEqual(result[3][courier_main.AUTO], "history")
        self.assertEqual(result[2][courier_main.AUTO], 4.0)

    def test_planning_percentile_is_computed_from_raw_grouped_orders(self):
        rows = pd.DataFrame(
            [
                {
                    "planning_location_id": "20",
                    "planning_location_name": "Donor",
                    "segment": "high",
                    "time_segment": "lunch",
                    "vehicle_type": "auto",
                    "duration_minutes": 10.0,
                },
                {
                    "planning_location_id": "20",
                    "planning_location_name": "Donor",
                    "segment": "high",
                    "time_segment": "lunch",
                    "vehicle_type": "auto",
                    "duration_minutes": 30.0,
                },
            ]
        )
        report = delivery_duration_report.build_planning_time_segment_report(rows)
        self.assertEqual(report.iloc[0]["orders_count"], 2)
        self.assertAlmostEqual(report.iloc[0]["p80_minutes"], 26.0)

    def test_sparse_location_falls_back_to_segment_cycle_time(self):
        duration = pd.DataFrame(
            [
                {
                    "planning_location_id": "1",
                    "segment": "high",
                    "time_segment": "lunch",
                    "vehicle_type": "auto",
                    "orders_count": 10,
                    "p80_minutes": 20,
                },
                {
                    "planning_location_id": "2",
                    "segment": "high",
                    "time_segment": "lunch",
                    "vehicle_type": "auto",
                    "orders_count": 50,
                    "p80_minutes": 30,
                },
            ]
        )
        location, segment, time_segment, global_vehicle = (
            courier_main.build_cycle_time_tables(duration)
        )
        row = {
            "location_id": "1",
            "segment": "high",
            "time_segment": "lunch",
        }
        value = courier_main.lookup_cycle_time(
            row,
            "auto",
            courier_main.make_duration_lookup(
                location,
                [
                    "planning_location_id",
                    "segment",
                    "time_segment",
                    "vehicle_type",
                ],
            ),
            courier_main.make_duration_lookup(
                segment,
                ["segment", "time_segment", "vehicle_type"],
            ),
            courier_main.make_duration_lookup(
                time_segment,
                ["time_segment", "vehicle_type"],
            ),
            courier_main.make_duration_lookup(
                global_vehicle,
                ["vehicle_type"],
            ),
        )
        self.assertEqual(value[2], "segment_cycle_time")
        self.assertAlmostEqual(value[0], (10 * 20 + 50 * 30) / 60)


if __name__ == "__main__":
    unittest.main()
