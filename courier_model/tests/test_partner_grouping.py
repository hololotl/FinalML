import tempfile
import unittest
from pathlib import Path

import pandas as pd

from courier_model import main as courier_main
from courier_model.partner_grouping import (
    build_partner_grouping,
    load_partner_donor_map,
    load_partner_grouping,
)


MAPPING_COLUMNS = [
    "partner_location_id",
    "partner_name",
    "kfm_donor_id",
    "kfm_donor_name",
    "mode",
    "notes",
]


def mapping_frame(rows):
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS).fillna("")


class PartnerGroupingTests(unittest.TestCase):
    def test_absorb_skip_and_unmapped(self):
        grouping = build_partner_grouping(
            mapping_frame(
                [
                    ["10", "Partner", "20", "Donor", "absorb", ""],
                    ["30", "Independent", "", "", "skip", ""],
                ]
            )
        )

        self.assertEqual(grouping.planning_location_id("10"), "20")
        self.assertEqual(grouping.planning_location_id("20"), "20")
        self.assertEqual(grouping.planning_location_id("30"), "30")
        self.assertEqual(grouping.planning_location_id("40"), "40")
        self.assertEqual(grouping.planning_to_members["20"], ["10", "20"])

    def test_transitive_mapping_and_cycle_validation(self):
        transitive = build_partner_grouping(
            mapping_frame(
                [
                    ["10", "A", "20", "B", "absorb", ""],
                    ["20", "B", "30", "C", "absorb", ""],
                ]
            )
        )
        self.assertEqual(transitive.planning_location_id("10"), "30")
        self.assertEqual(transitive.planning_location_id("20"), "30")

        with self.assertRaisesRegex(ValueError, "Cycle"):
            build_partner_grouping(
                mapping_frame(
                    [
                        ["10", "A", "20", "B", "absorb", ""],
                        ["20", "B", "10", "A", "absorb", ""],
                    ]
                )
            )

    def test_loader_rejects_conflicting_rows(self):
        frame = mapping_frame(
            [
                ["10", "Partner", "20", "Donor A", "absorb", ""],
                ["10", "Partner", "21", "Donor B", "absorb", ""],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_partner_donor_map(path)

    def test_prediction_orders_are_summed_before_slot_calculation(self):
        grouping = build_partner_grouping(
            mapping_frame(
                [
                    ["10", "Partner", "20", "Donor", "absorb", ""],
                    ["30", "Independent", "", "", "skip", ""],
                ]
            )
        )
        segment_datetime = pd.Timestamp("2026-07-20 12:00:00+03:00")
        predictions = pd.DataFrame(
            [
                {
                    "location_id": "10",
                    "segment_datetime": segment_datetime,
                    "segment": "medium",
                    "time_segment": "block_12_18",
                    "prediction": 2.4,
                    "orders_count": 2.0,
                },
                {
                    "location_id": "20",
                    "segment_datetime": segment_datetime,
                    "segment": "medium",
                    "time_segment": "block_12_18",
                    "prediction": 3.6,
                    "orders_count": 4.0,
                },
                {
                    "location_id": "30",
                    "segment_datetime": segment_datetime,
                    "segment": "medium",
                    "time_segment": "block_12_18",
                    "prediction": 1.0,
                    "orders_count": 1.0,
                },
            ]
        )

        grouped, audit = courier_main.aggregate_predictions_to_planning_locations(
            predictions,
            grouping,
        )

        donor = grouped[grouped["location_id"] == "20"].iloc[0]
        independent = grouped[grouped["location_id"] == "30"].iloc[0]
        self.assertAlmostEqual(donor["prediction"], 6.0)
        self.assertAlmostEqual(donor["orders_count"], 6.0)
        self.assertAlmostEqual(donor["absorbed_partner_orders"], 2.4)
        self.assertEqual(independent["prediction"], 1.0)
        self.assertEqual(audit.iloc[0]["source_location_id"], "10")
        self.assertAlmostEqual(grouped["prediction"].sum(), predictions["prediction"].sum())

    def test_legacy_group_ids_are_rejected(self):
        predictions = pd.DataFrame(
            [
                {
                    "location_id": "grp_10_20",
                    "segment_datetime": pd.Timestamp("2026-07-20 12:00:00+03:00"),
                    "segment": "medium",
                    "time_segment": "block_12_18",
                    "prediction": 1.0,
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "legacy grp_"):
            courier_main.aggregate_predictions_to_planning_locations(
                predictions,
                build_partner_grouping(mapping_frame([])),
            )

    def test_real_conflicting_legacy_pairs_now_have_distinct_donors(self):
        grouping = load_partner_grouping()
        self.assertEqual(grouping.planning_location_id("187"), "12")
        self.assertEqual(grouping.planning_location_id("296"), "101")
        self.assertEqual(grouping.planning_location_id("130"), "8")
        self.assertEqual(grouping.planning_location_id("139"), "30")


if __name__ == "__main__":
    unittest.main()
