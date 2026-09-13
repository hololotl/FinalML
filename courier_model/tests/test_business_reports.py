import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from courier_model import business_reports
from courier_model.orders_payment_report import load_orders_payment_report


class BusinessReportTests(unittest.TestCase):
    def test_segment_label_controls_canonical_hour_boundary(self):
        start = business_reports.canonical_segment_start_hour(
            "high",
            "early_morning",
            pd.Timestamp("2026-07-20 06:00:00+03:00"),
        )
        self.assertEqual(start, 8)

    def test_payment_report_parser_reads_metadata_and_rows(self):
        content = (
            "period_from,2026-07-01\n"
            "period_to,2026-08-01\n"
            "total_orders,2\n\n"
            "location_id,location_name,hour_from,hour_to,orders_count,"
            "avg_c_rate_total_rub,sum_c_rate_total_rub\n"
            "1,Test,8,9,2,300,600\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "payment.csv"
            path.write_text(content, encoding="utf-8")
            rows, metadata = load_orders_payment_report(path)

        self.assertEqual(metadata["period_from"], "2026-07-01")
        self.assertEqual(rows.iloc[0]["location_id"], "1")
        self.assertEqual(rows.iloc[0]["orders_count"], 2)

    def test_hourly_expansion_uses_historical_order_profile(self):
        forecast = pd.DataFrame(
            [
                {
                    "location_id": "1",
                    "segment_datetime": pd.Timestamp(
                        "2026-07-20 12:00:00+03:00"
                    ),
                    "segment": "medium",
                    "time_segment": "block_12_18",
                    "auto_order_prediction": 12.0,
                    "bike_order_prediction": 0.0,
                    "auto_slots_needed": 1,
                    "bike_slots_needed": 0,
                }
            ]
        )
        rates = pd.DataFrame(
            [
                {
                    "planning_location_id": "1",
                    "hour_from": hour,
                    "orders_count": 10 if hour == 12 else 2,
                    "sum_c_rate_total_rub": (
                        3000 if hour == 12 else 600
                    ),
                    "avg_c_rate_total_rub": 300,
                }
                for hour in range(12, 18)
            ]
        )
        hourly = business_reports.expand_to_hourly_demand(forecast, rates)
        auto = hourly[hourly["vehicle_type"] == "auto"]

        self.assertAlmostEqual(auto["predicted_orders"].sum(), 12.0)
        self.assertGreater(
            auto.loc[auto["hour"] == 12, "predicted_orders"].iloc[0],
            auto.loc[auto["hour"] == 13, "predicted_orders"].iloc[0],
        )
        self.assertAlmostEqual(
            auto["predicted_earning_pool_rub"].sum(),
            3600.0,
        )

    def test_long_shift_split_balances_earnings(self):
        values = np.array([100.0] * 5 + [10.0] * 10 + [0.0] * 9)
        chunks = business_reports.split_shift_into_max_hours(
            0,
            15,
            max_hours=12,
            hour_values=values,
        )
        self.assertEqual(chunks, [(0, 3), (3, 15)])
        self.assertTrue(all(finish - start <= 12 for start, finish in chunks))

    def test_single_courier_long_day_is_split_by_earnings(self):
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "1",
                    "date": "2026-07-20",
                    "hour": hour,
                    "segment": "medium",
                    "vehicle_type": "auto",
                    "slots_needed": 1,
                    "predicted_orders": 1.0,
                    "predicted_earning_pool_rub": 300.0,
                }
                for hour in range(8, 23)
            ]
        )
        plan = business_reports.build_business_shift_plan(
            hourly,
            {("1", 0): [(8.0, 23.0)]},
            {},
        )
        self.assertEqual(len(plan), 2)
        self.assertTrue((plan["shift_hours"] <= 12).all())
        self.assertAlmostEqual(
            plan["predicted_shift_earnings_rub"].sum(),
            4500,
        )

    def test_multi_courier_plan_covers_need_and_respects_maximum(self):
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "1",
                    "date": "2026-07-20",
                    "hour": hour,
                    "segment": "high",
                    "vehicle_type": "auto",
                    "slots_needed": 2,
                    "predicted_orders": 2.0,
                    "predicted_earning_pool_rub": (
                        800.0 if 12 <= hour < 16 else 200.0
                    ),
                }
                for hour in range(8, 23)
            ]
        )
        plan = business_reports.build_business_shift_plan(
            hourly,
            {("1", 0): [(8.0, 23.0)]},
            {},
        )
        self.assertTrue((plan["shift_hours"] <= 12).all())
        coverage = np.zeros(24, dtype=int)
        for row in plan.itertuples(index=False):
            start = int(row.shift_start.split(":")[0])
            finish = int(row.shift_finish.split(":")[0])
            coverage[start:finish] += int(row.slots_to_create)
        self.assertTrue((coverage[8:23] >= 2).all())
        validation = business_reports.build_control_week_validation(hourly, plan)
        metrics = dict(zip(validation["metric"], validation["value"]))
        self.assertEqual(metrics["coverage_shortfall_courier_hours"], 0)
        self.assertEqual(metrics["multi_courier_shifts_over_12h"], 0)

    def test_single_layer_trims_empty_open_window_edges(self):
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "1",
                    "date": "2026-07-20",
                    "hour": hour,
                    "segment": "medium",
                    "vehicle_type": "auto",
                    "slots_needed": 1 if 10 <= hour < 16 else 0,
                    "predicted_orders": 1.0 if 10 <= hour < 16 else 0.0,
                    "predicted_earning_pool_rub": (
                        300.0 if 10 <= hour < 16 else 0.0
                    ),
                }
                for hour in range(8, 23)
            ]
        )
        plan = business_reports.build_business_shift_plan(
            hourly,
            {("1", 0): [(8.0, 23.0)]},
            {},
        )
        self.assertEqual(plan.iloc[0]["shift_start"], "10:00")
        self.assertEqual(plan.iloc[0]["shift_finish"], "16:00")
        self.assertEqual(plan.iloc[0]["overcoverage_hours"], 0)

    def test_adjacent_short_prefix_is_merged_before_shift_split(self):
        merged_start, merged_finish, remaining = (
            business_reports._merge_adjacent_layer_runs(
                10,
                21,
                [(7, 10), (3, 5), (21, 22)],
            )
        )
        self.assertEqual((merged_start, merged_finish), (7, 22))
        self.assertEqual(remaining, [(3, 5)])

    def test_contiguous_layer_demand_does_not_create_short_gap_shifts(self):
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "29",
                    "date": "2026-09-14",
                    "hour": hour,
                    "segment": "mega",
                    "vehicle_type": "auto",
                    "slots_needed": need,
                    "predicted_orders": float(need),
                    "predicted_earning_pool_rub": float(need * 500),
                }
                for hour, need in zip(
                    range(7, 22),
                    [3, 5, 7, 7, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 2],
                )
            ]
        )
        plan = business_reports.build_business_shift_plan(
            hourly,
            {("29", 0): [(7.0, 22.0)]},
            {},
        )
        self.assertFalse(
            plan["shift_template"].str.fullmatch("demand_layer_gap").any()
        )
        self.assertTrue((plan["shift_hours"] <= 12).all())
        validation = business_reports.build_control_week_validation(
            hourly,
            plan,
            {("29", 0): [(7.0, 22.0)]},
            {},
        )
        metrics = dict(zip(validation["metric"], validation["value"]))
        self.assertEqual(metrics["coverage_shortfall_courier_hours"], 0)

    def test_calibration_gate_accepts_closer_candidate_with_full_coverage(self):
        baseline = pd.DataFrame(
            [{"vehicle_type": "auto", "shift_hours": 10, "slots_to_create": 2}]
        )
        candidate = pd.DataFrame(
            [{"vehicle_type": "auto", "shift_hours": 10, "slots_to_create": 1}]
        )
        validation = pd.DataFrame(
            [
                {"metric": "coverage_shortfall_courier_hours", "value": 0},
                {"metric": "all_shifts_over_12h", "value": 0},
            ]
        )
        comparison, accepted = (
            business_reports.build_courier_hour_calibration_comparison(
                baseline,
                candidate,
                {"auto": 9.0, "bike": 0.0},
                validation,
            )
        )
        self.assertTrue(accepted)
        self.assertTrue(comparison["candidate_accepted"].all())


if __name__ == "__main__":
    unittest.main()
