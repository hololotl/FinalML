import unittest

import pandas as pd

from courier_model import staffing_corridor as sc


def make_schedule(rows):
    frame = pd.DataFrame(rows)
    frame["schedule_start_dt"] = pd.to_datetime(frame["start"]).dt.tz_localize(
        "Europe/Moscow"
    )
    frame["schedule_finish_dt"] = pd.to_datetime(frame["finish"]).dt.tz_localize(
        "Europe/Moscow"
    )
    return frame.drop(columns=["start", "finish"])


class StaffingCorridorTests(unittest.TestCase):
    def test_schedule_hour_counts_uses_partial_hour_overlap(self):
        start = pd.Timestamp("2026-09-01 00:00", tz="Europe/Moscow")
        finish = pd.Timestamp("2026-09-02 00:00", tz="Europe/Moscow")
        schedule = make_schedule(
            [
                {
                    "location_id": "1",
                    "vehicle_type": "auto",
                    "start": "2026-09-01 07:30",
                    "finish": "2026-09-01 10:10",
                }
            ]
        )
        counts = sc.schedule_hour_counts(schedule, start, finish)
        # 07:30 counts (30 min overlap), 10:10 does not (10 min overlap)
        self.assertEqual(sorted(counts["hour"].tolist()), [7, 8, 9])

    def test_corridor_clamps_model_and_ignores_rare_cells(self):
        dates = pd.date_range("2026-09-01", periods=8)  # Tue..Tue, 2 weekend days
        weekday_dates = [d for d in dates if d.dayofweek < 5]
        rows = []
        for i, date in enumerate(weekday_dates):
            rows.append(
                {
                    "location_id": "1",
                    "date": date,
                    "hour": 12,
                    "vehicle_type": "auto",
                    "couriers": 3 if i % 2 == 0 else 4,
                }
            )
        # rare cell: one day only
        rows.append(
            {
                "location_id": "1",
                "date": weekday_dates[0],
                "hour": 3,
                "vehicle_type": "auto",
                "couriers": 5,
            }
        )
        corridor = sc.build_staffing_corridor(
            pd.DataFrame(rows),
            dates,
            low_quantile=0.25,
            high_quantile=0.75,
            min_active_share=0.5,
        )
        cell = corridor[(corridor["hour"] == 12) & (corridor["day_type"] == "weekday")]
        self.assertEqual(len(cell), 1)
        self.assertEqual(int(cell["corridor_lo"].iloc[0]), 3)
        self.assertEqual(int(cell["corridor_hi"].iloc[0]), 4)
        self.assertTrue(corridor[corridor["hour"] == 3].empty)

        hourly = pd.DataFrame(
            [
                {"location_id": "1", "date": "2026-09-08", "hour": 12,
                 "vehicle_type": "auto", "slots_needed": 8,
                 "predicted_orders": 10.0, "time_segment": "lunch"},
                {"location_id": "1", "date": "2026-09-08", "hour": 12,
                 "vehicle_type": "bike", "slots_needed": 2,
                 "predicted_orders": 10.0, "time_segment": "lunch"},
                {"location_id": "1", "date": "2026-09-08", "hour": 3,
                 "vehicle_type": "auto", "slots_needed": 1,
                 "predicted_orders": 1.0, "time_segment": "deep_night"},
            ]
        )
        result, audit = sc.apply_staffing_corridor(hourly, corridor)
        auto_noon = result[(result["hour"] == 12) & (result["vehicle_type"] == "auto")]
        self.assertEqual(int(auto_noon["slots_needed"].iloc[0]), 4)
        self.assertEqual(int(auto_noon["slots_needed_model"].iloc[0]), 8)
        # no corridor for bike or for the rare hour -> untouched
        bike_noon = result[(result["hour"] == 12) & (result["vehicle_type"] == "bike")]
        self.assertEqual(int(bike_noon["slots_needed"].iloc[0]), 2)
        night = result[result["hour"] == 3]
        self.assertEqual(int(night["slots_needed"].iloc[0]), 1)
        self.assertEqual(int(audit["slot_hours_lowered"].sum()), 4)


    def test_hour_staffing_gate_zeros_never_staffed_hours(self):
        dates = pd.date_range("2026-10-20", periods=7)
        rows = []
        for date in dates:
            rows.append(
                {
                    "location_id": "99",
                    "date": date,
                    "hour": 8,
                    "vehicle_type": "bike",
                    "couriers": 1,
                }
            )
        counts = pd.DataFrame(rows)
        active, locations = sc.build_recent_active_hours(counts)
        self.assertIn(("99", 8), active)
        self.assertNotIn(("99", 7), active)
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "99",
                    "date": "2026-10-27",
                    "hour": 7,
                    "vehicle_type": "bike",
                    "slots_needed": 2,
                    "slots_needed_model": 2,
                    "slot_demand_source": "model",
                },
                {
                    "location_id": "99",
                    "date": "2026-10-27",
                    "hour": 8,
                    "vehicle_type": "bike",
                    "slots_needed": 2,
                    "slots_needed_model": 2,
                    "slot_demand_source": "model",
                },
            ]
        )
        updated, audit = sc.apply_recent_hour_staffing_gate(
            hourly, active, locations
        )
        self.assertEqual(int(updated.loc[updated.hour == 7, "slots_needed"].iloc[0]), 0)
        self.assertEqual(int(updated.loc[updated.hour == 8, "slots_needed"].iloc[0]), 2)
        self.assertEqual(len(audit), 1)

    def test_feedback_closing_clamp_zeros_late_hours(self):
        hourly = pd.DataFrame(
            [
                {
                    "location_id": "115",
                    "date": "2026-11-02",
                    "hour": 21,
                    "vehicle_type": "auto",
                    "slots_needed": 1,
                },
                {
                    "location_id": "115",
                    "date": "2026-11-02",
                    "hour": 22,
                    "vehicle_type": "auto",
                    "slots_needed": 1,
                },
                {
                    "location_id": "39",
                    "date": "2026-11-02",
                    "hour": 20,
                    "vehicle_type": "bike",
                    "slots_needed": 2,
                },
                {
                    "location_id": "39",
                    "date": "2026-11-02",
                    "hour": 21,
                    "vehicle_type": "bike",
                    "slots_needed": 1,
                },
            ]
        )
        updated, audit = sc.apply_feedback_closing_hour_clamp(hourly)
        self.assertEqual(int(updated.loc[updated.hour == 21, "slots_needed"].iloc[0]), 1)
        self.assertEqual(int(updated.loc[updated.hour == 22, "slots_needed"].iloc[0]), 0)
        self.assertEqual(int(updated.loc[updated.location_id == "39", "slots_needed"].iloc[0]), 2)
        self.assertEqual(int(updated.loc[updated.location_id == "39", "slots_needed"].iloc[1]), 0)
        self.assertEqual(len(audit), 2)


if __name__ == "__main__":
    unittest.main()
