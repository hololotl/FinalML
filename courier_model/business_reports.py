import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from . import main as cm
except ImportError:
    import main as cm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FORECAST_PATH = SCRIPT_DIR / "res" / "courier_forecast.csv"
FORECAST_PATH = Path(os.getenv("COURIER_FORECAST_PATH", DEFAULT_FORECAST_PATH))
OUTPUT_DIR = Path(
    os.getenv(
        "BUSINESS_REPORT_OUTPUT_DIR",
        FORECAST_PATH.parent / "business_reports",
    )
)
BUSINESS_SHIFT_BUILD_MODE = os.getenv("BUSINESS_SHIFT_BUILD_MODE", "demand_layers")
BUSINESS_LAYER_ANCHOR_TO_OPEN = os.getenv("BUSINESS_LAYER_ANCHOR_TO_OPEN", "0") == "1"
BUSINESS_LAYER_USE_TEMPLATES = os.getenv("BUSINESS_LAYER_USE_TEMPLATES", "1") == "1"
BUSINESS_LAYER_FILL_GAPS = os.getenv("BUSINESS_LAYER_FILL_GAPS", "0") == "1"
MAX_SHIFT_HOURS = int(os.getenv("MAX_SHIFT_HOURS", "12"))
PARTNER_DONOR_MAP_PATH = Path(
    os.getenv(
        "PARTNER_DONOR_MAP_PATH",
        SCRIPT_DIR / "partner_donor_map.csv",
    )
)
KFM_ORGANIZATION_IDS = {
    "5",
    "3",
    "12",
    "23",
    "2",
    "16",
    "13",
    "10",
    "15",
    "9",
    "1",
    "8",
    "30",
    "7",
    "147",
}

SHIFT_TEMPLATES = [
    ("night_00_08", 0, 8),
    ("early_06_14", 6, 8),
    ("morning_08_16", 8, 8),
    ("day_08_18", 8, 10),
    ("mid_10_20", 10, 10),
    ("day_evening_12_22", 12, 10),
    ("evening_14_24", 14, 10),
    ("late_16_24", 16, 8),
]


def load_forecast(path):
    df = pd.read_csv(path)
    df["segment_datetime"] = pd.to_datetime(
        df["segment_datetime"],
        errors="coerce",
    )
    df = df.dropna(subset=["segment_datetime"])
    return df


def load_locations_metadata():
    engine = cm.build_engine()
    locations_df = pd.read_sql_query(
        text(
            "SELECT id AS location_id, name AS location_name, "
            "organization AS organization_id, COALESCE(transport, 0) AS transport "
            "FROM locations"
        ),
        engine,
    )
    locations_df["location_id"] = locations_df["location_id"].astype(str)
    locations_df["organization_id"] = locations_df["organization_id"].astype(str)
    locations_df["transport"] = locations_df["transport"].astype(int)
    return locations_df


def build_location_transport_lookup(locations_df):
    return {
        str(row.location_id): int(row.transport)
        for row in locations_df.itertuples(index=False)
    }


def load_work_hours():
    engine = cm.build_engine()
    work_hours_df = pd.read_sql_query(
        text(
            "SELECT location_id, weekday, working, start_hour, start_minutes, "
            "finish_hour, finish_minutes FROM work_hours WHERE working = true"
        ),
        engine,
    )
    if work_hours_df.empty:
        return work_hours_df
    work_hours_df["location_id"] = work_hours_df["location_id"].astype(str)
    return work_hours_df


def build_location_name_map(forecast_df, locations_df):
    location_names = dict(
        zip(locations_df["location_id"], locations_df["location_name"])
    )
    group_to_members = build_group_to_members()
    result = {}
    for location_id in forecast_df["location_id"].astype(str).unique():
        if location_id.startswith("grp_"):
            member_names = [
                location_names.get(member_id, member_id)
                for member_id in sorted(
                    group_to_members.get(location_id, []),
                    key=lambda value: int(value) if str(value).isdigit() else 0,
                )
            ]
            result[location_id] = " + ".join(member_names) if member_names else location_id
        else:
            result[location_id] = location_names.get(location_id, location_id)
    return result


def build_business_group_map(forecast_df, locations_df):
    organization_by_location = dict(
        zip(locations_df["location_id"], locations_df["organization_id"])
    )
    group_to_members = build_group_to_members()

    result = {}
    for location_id in forecast_df["location_id"].astype(str).unique():
        if location_id.startswith("grp_"):
            member_ids = group_to_members.get(location_id, [])
            member_is_kfm = [
                organization_by_location.get(member_id) in KFM_ORGANIZATION_IDS
                for member_id in member_ids
            ]
            if member_is_kfm and all(member_is_kfm):
                result[location_id] = "kfm"
            elif member_is_kfm and not any(member_is_kfm):
                result[location_id] = "non_kfm"
            else:
                result[location_id] = "mixed"
        else:
            organization_id = organization_by_location.get(location_id)
            result[location_id] = (
                "kfm" if organization_id in KFM_ORGANIZATION_IDS else "non_kfm"
            )
    return result


def build_group_to_members():
    location_to_group = cm.build_merged_location_groups()
    group_to_members = {}
    for member_id, group_id in location_to_group.items():
        group_to_members.setdefault(group_id, []).append(str(member_id))
    return group_to_members


def load_partner_donor_map(path=PARTNER_DONOR_MAP_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "partner_location_id",
                "partner_name",
                "kfm_donor_id",
                "kfm_donor_name",
                "mode",
                "notes",
            ]
        )
    mapping = pd.read_csv(path, dtype=str).fillna("")
    mapping["partner_location_id"] = mapping["partner_location_id"].astype(str)
    mapping["kfm_donor_id"] = mapping["kfm_donor_id"].astype(str)
    mapping["mode"] = mapping["mode"].str.strip().str.lower()
    return mapping


def _group_members_from_id(location_id, group_to_members):
    location_id = str(location_id)
    members = [str(m) for m in group_to_members.get(location_id, [])]
    if members:
        return members
    if location_id.startswith("grp_"):
        return [part for part in location_id.replace("grp_", "").split("_") if part]
    return []


def resolve_absorb_target(location_id, absorb_by_partner, group_to_members):
    """Return (action, donor_id, partner_ids) for a forecast location_id."""
    location_id = str(location_id)
    if location_id in absorb_by_partner:
        return "absorb", absorb_by_partner[location_id], [location_id]

    if not location_id.startswith("grp_"):
        return "keep", None, []

    members = _group_members_from_id(location_id, group_to_members)
    if not members:
        return "keep", None, []

    absorb_members = [m for m in members if m in absorb_by_partner]
    if not absorb_members:
        return "keep", None, members
    # Draft rule: absorb whole merged group only if every member is mapped to absorb.
    if len(absorb_members) != len(members):
        return "keep", None, members

    donors = [absorb_by_partner[m] for m in absorb_members]
    donor_id = donors[0]
    return "absorb", donor_id, absorb_members


def absorb_partner_demand(forecast_df, mapping_df, group_to_members):
    """Move partner slot demand onto KFM donors; drop absorbed partners from publish set."""
    forecast = forecast_df.copy()
    forecast["location_id"] = forecast["location_id"].astype(str)
    forecast["segment_datetime"] = pd.to_datetime(forecast["segment_datetime"])

    for col, default in [
        ("absorbed_partner_auto_slots", 0),
        ("absorbed_partner_bike_slots", 0),
        ("absorbed_partners", ""),
    ]:
        if col not in forecast.columns:
            forecast[col] = default

    absorb_rows = mapping_df[mapping_df["mode"] == "absorb"].copy()
    absorb_rows = absorb_rows[absorb_rows["kfm_donor_id"].astype(str).str.len() > 0]
    absorb_by_partner = {
        str(row.partner_location_id): str(row.kfm_donor_id)
        for row in absorb_rows.itertuples(index=False)
    }
    partner_name_by_id = {
        str(row.partner_location_id): str(row.partner_name)
        for row in mapping_df.itertuples(index=False)
    }

    empty_audit = pd.DataFrame(
        columns=[
            "source_location_id",
            "partner_ids",
            "partner_names",
            "kfm_donor_id",
            "segment_datetime",
            "time_segment",
            "auto_slots_moved",
            "bike_slots_moved",
            "action",
        ]
    )
    if not absorb_by_partner:
        return forecast, empty_audit

    targets = {}
    for location_id in forecast["location_id"].unique():
        action, donor_id, partner_ids = resolve_absorb_target(
            location_id,
            absorb_by_partner,
            group_to_members,
        )
        if action == "absorb":
            targets[str(location_id)] = (donor_id, partner_ids)

    if not targets:
        return forecast, empty_audit

    audit_rows = []
    # key -> {auto, bike, partners, template_row}
    donor_additions = {}
    drop_locations = set()

    for source_location_id, (donor_id, partner_ids) in targets.items():
        source_mask = forecast["location_id"] == source_location_id
        source_rows = forecast.loc[source_mask]
        if source_rows.empty:
            continue
        drop_locations.add(source_location_id)
        partner_names = ",".join(
            partner_name_by_id.get(pid, pid) for pid in partner_ids
        )
        for _, prow in source_rows.iterrows():
            auto_slots = int(prow.get("auto_slots_needed", 0) or 0)
            bike_slots = int(prow.get("bike_slots_needed", 0) or 0)
            key = (
                donor_id,
                pd.Timestamp(prow["segment_datetime"]),
                str(prow["time_segment"]),
            )
            current = donor_additions.get(key)
            if current is None:
                template = prow.copy()
                template["location_id"] = donor_id
                template["absorbed_partner_auto_slots"] = 0
                template["absorbed_partner_bike_slots"] = 0
                template["absorbed_partners"] = ""
                donor_additions[key] = {
                    "auto": auto_slots,
                    "bike": bike_slots,
                    "partners": partner_names,
                    "template": template,
                }
            else:
                current["auto"] += auto_slots
                current["bike"] += bike_slots
                merged = ",".join(
                    dict.fromkeys(
                        part
                        for part in (
                            current["partners"].split(",") + partner_names.split(",")
                        )
                        if part
                    )
                )
                current["partners"] = merged

            audit_rows.append({
                "source_location_id": source_location_id,
                "partner_ids": ",".join(partner_ids),
                "partner_names": partner_names,
                "kfm_donor_id": donor_id,
                "segment_datetime": prow["segment_datetime"],
                "time_segment": prow["time_segment"],
                "auto_slots_moved": auto_slots,
                "bike_slots_moved": bike_slots,
                "action": "absorb",
            })

    forecast = forecast.loc[~forecast["location_id"].isin(drop_locations)].copy()
    forecast = forecast.reset_index(drop=True)

    lookup = {
        (
            str(row.location_id),
            pd.Timestamp(row.segment_datetime),
            str(row.time_segment),
        ): idx
        for idx, row in forecast.iterrows()
    }

    new_rows = []
    for key, payload in donor_additions.items():
        donor_id, segment_datetime, time_segment = key
        auto_add = int(payload["auto"])
        bike_add = int(payload["bike"])
        partners_text = payload["partners"]
        if key in lookup:
            idx = lookup[key]
            forecast.at[idx, "auto_slots_needed"] = (
                int(forecast.at[idx, "auto_slots_needed"] or 0) + auto_add
            )
            forecast.at[idx, "bike_slots_needed"] = (
                int(forecast.at[idx, "bike_slots_needed"] or 0) + bike_add
            )
            forecast.at[idx, "total_slots_needed"] = (
                int(forecast.at[idx, "auto_slots_needed"])
                + int(forecast.at[idx, "bike_slots_needed"])
            )
            for alias_src, alias_dst in [
                ("auto_slots_needed", "auto_couriers_needed"),
                ("bike_slots_needed", "bike_couriers_needed"),
                ("total_slots_needed", "total_couriers_needed"),
            ]:
                if alias_dst in forecast.columns:
                    forecast.at[idx, alias_dst] = forecast.at[idx, alias_src]
            forecast.at[idx, "absorbed_partner_auto_slots"] = (
                int(forecast.at[idx, "absorbed_partner_auto_slots"] or 0) + auto_add
            )
            forecast.at[idx, "absorbed_partner_bike_slots"] = (
                int(forecast.at[idx, "absorbed_partner_bike_slots"] or 0) + bike_add
            )
            prev = str(forecast.at[idx, "absorbed_partners"] or "")
            forecast.at[idx, "absorbed_partners"] = ",".join(
                dict.fromkeys(
                    part for part in (prev.split(",") + partners_text.split(",")) if part
                )
            )
        else:
            new_row = payload["template"].copy()
            donor_segment = forecast.loc[forecast["location_id"] == donor_id, "segment"]
            if not donor_segment.empty:
                new_row["segment"] = donor_segment.iloc[0]
            new_row["location_id"] = donor_id
            new_row["segment_datetime"] = segment_datetime
            new_row["time_segment"] = time_segment
            new_row["auto_slots_needed"] = auto_add
            new_row["bike_slots_needed"] = bike_add
            new_row["total_slots_needed"] = auto_add + bike_add
            new_row["auto_couriers_needed"] = auto_add
            new_row["bike_couriers_needed"] = bike_add
            new_row["total_couriers_needed"] = auto_add + bike_add
            new_row["orders_prediction"] = 0.0
            new_row["auto_order_prediction"] = 0.0
            new_row["bike_order_prediction"] = 0.0
            new_row["absorbed_partner_auto_slots"] = auto_add
            new_row["absorbed_partner_bike_slots"] = bike_add
            new_row["absorbed_partners"] = partners_text
            new_rows.append(new_row)

    if new_rows:
        forecast = pd.concat([forecast, pd.DataFrame(new_rows)], ignore_index=True)

    return forecast, pd.DataFrame(audit_rows)


def build_partner_absorb_summary(audit_df, location_name_map):
    if audit_df is None or audit_df.empty:
        return pd.DataFrame(
            columns=[
                "kfm_donor_id",
                "kfm_donor_name",
                "partner_ids",
                "partner_names",
                "auto_slots_moved",
                "bike_slots_moved",
                "total_slots_moved",
            ]
        )
    summary = (
        audit_df.groupby(["kfm_donor_id"], as_index=False)
        .agg(
            partner_ids=(
                "partner_ids",
                lambda s: ",".join(
                    dict.fromkeys(
                        pid for cell in s for pid in str(cell).split(",") if pid
                    )
                ),
            ),
            partner_names=(
                "partner_names",
                lambda s: ",".join(
                    dict.fromkeys(
                        name for cell in s for name in str(cell).split(",") if name
                    )
                ),
            ),
            auto_slots_moved=("auto_slots_moved", "sum"),
            bike_slots_moved=("bike_slots_moved", "sum"),
        )
    )
    summary["total_slots_moved"] = (
        summary["auto_slots_moved"] + summary["bike_slots_moved"]
    )
    summary["kfm_donor_name"] = summary["kfm_donor_id"].map(
        lambda x: location_name_map.get(str(x), str(x))
    )
    return summary.sort_values("total_slots_moved", ascending=False)


def _hour_value(hour, minutes):
    if pd.isna(hour):
        return np.nan
    minutes = 0 if pd.isna(minutes) else minutes
    return float(hour) + float(minutes) / 60.0


def _window_to_intervals(start_hour, finish_hour):
    """Expand a DB work window into same-calendar-day intervals.

    Conventions from work_hours:
    - start < finish → same-day window, e.g. 08-23
    - start > finish and finish == 0 → open until end of day, e.g. 07-0 → 07:00-24:00
    - start > finish and finish > 0 → overnight, e.g. 23-7 → 23:00-24:00 and 00:00-07:00
    """
    if pd.isna(start_hour) or pd.isna(finish_hour):
        return []
    if start_hour == finish_hour:
        return [(0.0, 24.0)]
    if start_hour < finish_hour:
        return [(float(start_hour), float(finish_hour))]
    # finish == 0 means "until midnight", not "until 00:00 next day".
    if float(finish_hour) == 0.0:
        return [(float(start_hour), 24.0)]
    intervals = []
    if finish_hour > 0:
        intervals.append((0.0, float(finish_hour)))
    if start_hour < 24:
        intervals.append((float(start_hour), 24.0))
    return intervals


def _format_hour_value(value):
    hour = int(value)
    minute = int(round((float(value) - hour) * 60))
    if minute == 60:
        hour += 1
        minute = 0
    return f"{hour:02d}:{minute:02d}"


def _overnight_display_window(intervals):
    """If intervals are an overnight pair (0-F + S-24), return (S, F) for display."""
    pieces = [(start, finish) for start, finish in intervals if finish > start]
    if len(pieces) != 2:
        return None
    ordered = sorted(pieces)
    morning_start, morning_finish = ordered[0]
    evening_start, evening_finish = ordered[1]
    if morning_start != 0.0 or evening_finish < 24.0:
        return None
    if evening_start <= morning_finish:
        return None
    return evening_start, morning_finish


def format_intervals(intervals):
    overnight = _overnight_display_window(intervals)
    if overnight is not None:
        start, finish = overnight
        return f"{_format_hour_value(start)}-{_format_hour_value(finish)}"
    return ";".join(
        f"{_format_hour_value(start)}-{_format_hour_value(finish)}"
        for start, finish in intervals
        if finish > start
    )


def _merge_intervals(intervals):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, finish in ordered[1:]:
        prev_start, prev_finish = merged[-1]
        if start <= prev_finish:
            merged[-1] = (prev_start, max(prev_finish, finish))
        else:
            merged.append((start, finish))
    return merged


def build_work_interval_lookup(work_hours_df):
    lookup = {}
    if work_hours_df is None or work_hours_df.empty:
        return lookup

    for row in work_hours_df.itertuples(index=False):
        start_hour = _hour_value(row.start_hour, row.start_minutes)
        finish_hour = _hour_value(row.finish_hour, row.finish_minutes)
        lookup[(str(row.location_id), int(row.weekday))] = _window_to_intervals(
            start_hour,
            finish_hour,
        )
    return lookup


def get_open_intervals(location_id, weekday, work_interval_lookup, group_to_members):
    location_id = str(location_id)
    if location_id.startswith("grp_"):
        intervals = []
        for member_id in group_to_members.get(location_id, []):
            intervals.extend(work_interval_lookup.get((member_id, weekday), []))
        return _merge_intervals(intervals)
    return work_interval_lookup.get((location_id, weekday), [])


def template_fits_open_intervals(start_hour, end_hour, open_intervals):
    if not open_intervals:
        return True
    return any(
        start_hour >= interval_start and end_hour <= interval_finish
        for interval_start, interval_finish in open_intervals
    )


def hour_is_inside_open_intervals(hour, open_intervals):
    if not open_intervals:
        return True
    return any(
        hour >= interval_start and (hour + 1) <= interval_finish
        for interval_start, interval_finish in open_intervals
    )


def build_work_window_report(forecast_df, work_interval_lookup, group_to_members):
    rows = []
    date_frame = (
        forecast_df[["location_id", "segment_datetime"]]
        .drop_duplicates()
        .copy()
    )
    date_frame["date"] = date_frame["segment_datetime"].dt.date.astype(str)
    date_frame["weekday"] = date_frame["segment_datetime"].dt.dayofweek
    date_frame = date_frame[["location_id", "date", "weekday"]].drop_duplicates()
    for row in date_frame.itertuples(index=False):
        intervals = get_open_intervals(
            row.location_id,
            int(row.weekday),
            work_interval_lookup,
            group_to_members,
        )
        rows.append({
            "location_id": str(row.location_id),
            "date": row.date,
            "weekday": int(row.weekday),
            "open_intervals": format_intervals(intervals),
            "has_work_hours": bool(intervals),
        })
    return pd.DataFrame(rows)


def canonical_segment_start_hour(segment, time_segment, segment_datetime):
    start_map = dict(cm.segment_start_hours(segment))
    if time_segment in start_map:
        return int(start_map[time_segment])
    return int(pd.Timestamp(segment_datetime).hour)


def expand_to_hourly_demand(forecast_df):
    rows = []
    for row in forecast_df.itertuples(index=False):
        segment_start = pd.Timestamp(row.segment_datetime)
        hours = int(round(cm.segment_hours(row.time_segment)))
        if hours <= 0:
            continue

        start_hour = canonical_segment_start_hour(
            row.segment,
            row.time_segment,
            segment_start,
        )
        business_date = segment_start.date().isoformat()
        for hour_offset in range(hours):
            hour = start_hour + hour_offset
            if hour < 0 or hour >= 24:
                continue
            rows.append({
                "location_id": str(row.location_id),
                "date": business_date,
                "hour": hour,
                "segment": row.segment,
                "time_segment": row.time_segment,
                "vehicle_type": cm.AUTO,
                "slots_needed": int(row.auto_slots_needed),
                "predicted_orders": float(row.auto_order_prediction) / hours,
            })
            rows.append({
                "location_id": str(row.location_id),
                "date": business_date,
                "hour": hour,
                "segment": row.segment,
                "time_segment": row.time_segment,
                "vehicle_type": cm.BIKE,
                "slots_needed": int(row.bike_slots_needed),
                "predicted_orders": float(row.bike_order_prediction) / hours,
            })

    hourly = pd.DataFrame(rows)
    if hourly.empty:
        return hourly

    return (
        hourly.groupby(
            ["location_id", "date", "hour", "segment", "vehicle_type"],
            as_index=False,
        )
        .agg(
            slots_needed=("slots_needed", "max"),
            predicted_orders=("predicted_orders", "sum"),
        )
    )


def build_hour_arrays(group):
    demand = np.zeros(24, dtype=int)
    orders = np.zeros(24, dtype=float)
    for row in group.itertuples(index=False):
        demand[int(row.hour)] = max(demand[int(row.hour)], int(row.slots_needed))
        orders[int(row.hour)] += float(row.predicted_orders)
    return demand, orders


def build_single_courier_open_shifts(
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    open_intervals,
):
    rows = []
    candidate_intervals = open_intervals or [(0.0, 24.0)]
    for interval_start, interval_finish in candidate_intervals:
        start_bound = int(np.ceil(interval_start))
        finish_bound = int(np.floor(interval_finish))
        if finish_bound <= start_bound:
            continue

        active_hours = int((demand[start_bound:finish_bound] > 0).sum())
        duration_hours = finish_bound - start_bound
        rows.append({
            "location_id": location_id,
            "date": date,
            "segment": segment,
            "vehicle_type": vehicle_type,
            "shift_template": "full_open_window",
            "shift_start": f"{start_bound:02d}:00",
            "shift_finish": f"{finish_bound:02d}:00",
            "shift_hours": duration_hours,
            "open_intervals": format_intervals(candidate_intervals),
            "slots_to_create": 1,
            "covered_need_hours": active_hours,
            "template_hours": duration_hours,
            "overcoverage_hours": duration_hours - active_hours,
            "covered_predicted_orders": float(
                hourly_orders[start_bound:finish_bound].sum()
            ),
        })
    return rows


def choose_shift_template(residual, open_intervals):
    best = None
    for template_name, start_hour, duration_hours in SHIFT_TEMPLATES:
        if duration_hours > MAX_SHIFT_HOURS:
            continue
        end_hour = start_hour + duration_hours
        if not template_fits_open_intervals(start_hour, end_hour, open_intervals):
            continue
        covered = residual[start_hour:end_hour].clip(min=0).sum()
        if covered <= 0:
            continue
        over_hours = duration_hours - int((residual[start_hour:end_hour] > 0).sum())
        score = (covered / duration_hours, covered, -over_hours, duration_hours)
        candidate = (score, template_name, start_hour, end_hour, duration_hours)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    _, template_name, start_hour, end_hour, duration_hours = best
    return template_name, start_hour, end_hour, duration_hours


def choose_fallback_shift(residual, open_intervals):
    active_hours = np.where(residual > 0)[0]
    if len(active_hours) == 0:
        return None

    candidate_intervals = open_intervals or [(0.0, 24.0)]
    best = None
    for interval_start, interval_finish in candidate_intervals:
        start_hour = int(max(np.ceil(interval_start), active_hours.min()))
        end_hour = int(min(np.floor(interval_finish), active_hours.max() + 1))
        if end_hour <= start_hour:
            continue
        # Cap fallback windows at MAX_SHIFT_HOURS.
        end_hour = min(end_hour, start_hour + MAX_SHIFT_HOURS)
        covered = residual[start_hour:end_hour].clip(min=0).sum()
        if covered <= 0:
            continue
        duration_hours = end_hour - start_hour
        score = (covered / duration_hours, covered, -duration_hours)
        candidate = (score, "custom_open_window", start_hour, end_hour, duration_hours)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    _, template_name, start_hour, end_hour, duration_hours = best
    return template_name, start_hour, end_hour, duration_hours


def split_shift_into_max_hours(shift_start, shift_finish, max_hours=MAX_SHIFT_HOURS):
    """
    Split [shift_start, shift_finish) into chunks of length <= max_hours,
    preferring near-equal lengths instead of max-fill leftovers.

    Example: 15h with max=12 -> [8, 7], not [12, 3].
    """
    start = int(shift_start)
    finish = int(shift_finish)
    total_hours = finish - start
    if total_hours <= 0:
        return []
    if total_hours <= max_hours:
        return [(start, finish)]

    chunk_count = int(np.ceil(total_hours / max_hours))
    base = total_hours // chunk_count
    remainder = total_hours % chunk_count

    chunks = []
    current = start
    for index in range(chunk_count):
        # Put the longer pieces first so early day gets slightly longer coverage.
        duration = base + (1 if index < remainder else 0)
        chunk_finish = current + duration
        chunks.append((current, chunk_finish))
        current = chunk_finish
    return chunks


def build_shift_plan_for_group(
    location_id,
    date,
    segment,
    vehicle_type,
    group,
    open_intervals,
):
    residual, hourly_orders = build_hour_arrays(group)
    for hour in range(24):
        if not hour_is_inside_open_intervals(hour, open_intervals):
            residual[hour] = 0
            hourly_orders[hour] = 0

    if int(residual.max()) == 1:
        return build_single_courier_open_shifts(
            location_id,
            date,
            segment,
            vehicle_type,
            residual,
            hourly_orders,
            open_intervals,
        )

    if BUSINESS_SHIFT_BUILD_MODE == "demand_layers":
        return build_layered_shift_plan_for_group(
            location_id,
            date,
            segment,
            vehicle_type,
            residual,
            hourly_orders,
            open_intervals,
        )

    rows = []

    while residual.max() > 0:
        chosen = choose_shift_template(residual, open_intervals)
        if chosen is None:
            chosen = choose_fallback_shift(residual, open_intervals)
        if chosen is None:
            break
        template_name, start_hour, end_hour, duration_hours = chosen
        active_hours = residual[start_hour:end_hour] > 0
        covered_need_hours = int(active_hours.sum())
        covered_order_sum = float(hourly_orders[start_hour:end_hour].sum())

        residual[start_hour:end_hour] = np.maximum(
            residual[start_hour:end_hour] - 1,
            0,
        )
        rows.append({
            "location_id": location_id,
            "date": date,
            "segment": segment,
            "vehicle_type": vehicle_type,
            "shift_template": template_name,
            "shift_start": f"{start_hour:02d}:00",
            "shift_finish": f"{end_hour:02d}:00",
            "shift_hours": duration_hours,
            "open_intervals": format_intervals(open_intervals),
            "slots_to_create": 1,
            "covered_need_hours": covered_need_hours,
            "template_hours": duration_hours,
            "overcoverage_hours": duration_hours - covered_need_hours,
            "covered_predicted_orders": covered_order_sum,
        })

    return rows


def _layer_active_finish_hour(demand, start_bound, finish_bound, layer):
    layer_slice = demand[start_bound:finish_bound] >= layer
    if not layer_slice.any():
        return start_bound
    return start_bound + int(np.where(layer_slice)[0].max()) + 1


def _append_shift_row(
    rows,
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    open_intervals_display,
    shift_start,
    shift_finish,
    layer,
    template_name,
    enforce_max_shift_hours=True,
):
    if shift_finish <= shift_start:
        return
    if enforce_max_shift_hours:
        chunks = split_shift_into_max_hours(
            shift_start,
            shift_finish,
            MAX_SHIFT_HOURS,
        )
    else:
        chunks = [(int(shift_start), int(shift_finish))]
    for chunk_start, chunk_finish in chunks:
        active_hours = int((demand[chunk_start:chunk_finish] >= layer).sum())
        if active_hours <= 0:
            continue
        duration_hours = chunk_finish - chunk_start
        rows.append({
            "location_id": location_id,
            "date": date,
            "segment": segment,
            "vehicle_type": vehicle_type,
            "shift_template": template_name,
            "shift_start": f"{chunk_start:02d}:00",
            "shift_finish": f"{chunk_finish:02d}:00",
            "shift_hours": duration_hours,
            "open_intervals": format_intervals(open_intervals_display),
            "slots_to_create": 1,
            "covered_need_hours": active_hours,
            "template_hours": duration_hours,
            "overcoverage_hours": duration_hours - active_hours,
            "covered_predicted_orders": float(
                hourly_orders[chunk_start:chunk_finish].sum()
            ),
        })


def _iter_contiguous_true_runs(mask):
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def _build_layer_rows_for_interval(
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    open_interval,
    open_intervals_display,
):
    interval_start, interval_finish = open_interval
    start_bound = int(np.ceil(interval_start))
    finish_bound = int(np.floor(interval_finish))
    rows = []
    if finish_bound <= start_bound:
        return rows

    max_slots = int(demand[start_bound:finish_bound].max())
    for layer in range(1, max_slots + 1):
        layer_mask = demand[start_bound:finish_bound] >= layer
        if not layer_mask.any():
            continue

        if BUSINESS_LAYER_USE_TEMPLATES:
            active_finish = _layer_active_finish_hour(
                demand,
                start_bound,
                finish_bound,
                layer,
            )
            if layer == 1:
                shift_start = start_bound
                shift_finish = finish_bound
                template_name = "full_open_window"
                enforce_max_shift_hours = True
            else:
                residual = np.zeros(24, dtype=int)
                residual[start_bound:finish_bound] = layer_mask.astype(int)
                chosen = choose_shift_template(residual, [open_interval])
                if chosen is None:
                    chosen = choose_fallback_shift(residual, [open_interval])
                if chosen is None:
                    continue
                template_name, shift_start, shift_finish, _ = chosen
                shift_finish = max(int(shift_finish), int(active_finish))
                shift_finish = min(int(shift_finish), int(finish_bound))
                enforce_max_shift_hours = True
            _append_shift_row(
                rows,
                location_id,
                date,
                segment,
                vehicle_type,
                demand,
                hourly_orders,
                open_intervals_display,
                shift_start,
                shift_finish,
                layer,
                template_name,
                enforce_max_shift_hours=enforce_max_shift_hours,
            )
            continue

        for run_start, run_finish in _iter_contiguous_true_runs(layer_mask):
            shift_start = start_bound + int(run_start)
            shift_finish = start_bound + int(run_finish)

            if BUSINESS_LAYER_ANCHOR_TO_OPEN and run_start == 0:
                shift_start = start_bound
            if interval_finish - shift_finish <= 1:
                shift_finish = finish_bound

            _append_shift_row(
                rows,
                location_id,
                date,
                segment,
                vehicle_type,
                demand,
                hourly_orders,
                open_intervals_display,
                shift_start,
                shift_finish,
                layer,
                "demand_layer",
            )

    return rows


def _fill_shift_coverage_gaps(
    rows,
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    open_intervals,
):
    if not rows:
        return rows

    coverage = np.zeros(24, dtype=int)
    for row in rows:
        shift_start = int(str(row["shift_start"]).split(":")[0])
        shift_finish = int(str(row["shift_finish"]).split(":")[0])
        coverage[shift_start:shift_finish] += 1

    for interval_start, interval_finish in open_intervals or [(0.0, 24.0)]:
        start_bound = int(np.ceil(interval_start))
        finish_bound = int(np.floor(interval_finish))
        if finish_bound <= start_bound:
            continue

        residual = np.maximum(
            demand[start_bound:finish_bound] - coverage[start_bound:finish_bound],
            0,
        )
        while residual.max() > 0:
            padded = np.zeros(24, dtype=int)
            padded[start_bound:finish_bound] = residual
            chosen = choose_fallback_shift(
                padded,
                [(interval_start, interval_finish)],
            )
            if chosen is None:
                break
            template_name, shift_start, shift_finish, _ = chosen
            layer = int(residual.max())
            _append_shift_row(
                rows,
                location_id,
                date,
                segment,
                vehicle_type,
                demand,
                hourly_orders,
                open_intervals,
                shift_start,
                shift_finish,
                layer,
                template_name,
            )
            slice_start = shift_start - start_bound
            slice_finish = shift_finish - start_bound
            residual[slice_start:slice_finish] = np.maximum(
                residual[slice_start:slice_finish] - 1,
                0,
            )
            coverage[shift_start:shift_finish] += 1

    return rows


def build_layered_shift_plan_for_group(
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    open_intervals,
):
    rows = []
    candidate_intervals = open_intervals or [(0.0, 24.0)]
    for open_interval in candidate_intervals:
        rows.extend(
            _build_layer_rows_for_interval(
                location_id,
                date,
                segment,
                vehicle_type,
                demand,
                hourly_orders,
                open_interval,
                candidate_intervals,
            )
        )
    if BUSINESS_LAYER_USE_TEMPLATES and BUSINESS_LAYER_FILL_GAPS:
        rows = _fill_shift_coverage_gaps(
            rows,
            location_id,
            date,
            segment,
            vehicle_type,
            demand,
            hourly_orders,
            candidate_intervals,
        )
    return rows


def build_business_shift_plan(hourly_df, work_interval_lookup, group_to_members):
    rows = []
    if hourly_df.empty:
        return pd.DataFrame()

    group_cols = ["location_id", "date", "segment", "vehicle_type"]
    for key, group in hourly_df.groupby(group_cols):
        location_id, date, segment, vehicle_type = key
        rows.extend(
            build_shift_plan_for_group(
                location_id,
                date,
                segment,
                vehicle_type,
                group,
                get_open_intervals(
                    location_id,
                    pd.Timestamp(date).dayofweek,
                    work_interval_lookup,
                    group_to_members,
                ),
            )
        )

    if not rows:
        return pd.DataFrame()

    shift_plan = pd.DataFrame(rows)
    return (
        shift_plan.groupby(
            [
                "location_id",
                "date",
                "segment",
                "vehicle_type",
                "shift_template",
                "shift_start",
                "shift_finish",
                "shift_hours",
                "open_intervals",
            ],
            as_index=False,
        )
        .agg(
            slots_to_create=("slots_to_create", "sum"),
            covered_need_hours=("covered_need_hours", "sum"),
            template_hours=("template_hours", "sum"),
            overcoverage_hours=("overcoverage_hours", "sum"),
            shift_window_predicted_orders=("covered_predicted_orders", "max"),
        )
        .sort_values(
            ["date", "location_id", "vehicle_type", "shift_start", "shift_finish"]
        )
    )


def build_reports(
    forecast_df,
    shift_plan,
    location_name_map,
    business_group_map,
    work_window_report,
):
    forecast_df = forecast_df.copy()
    forecast_df["date"] = forecast_df["segment_datetime"].dt.date.astype(str)
    forecast_df["location_name"] = (
        forecast_df["location_id"].astype(str).map(location_name_map)
    )
    forecast_df["business_group"] = (
        forecast_df["location_id"].astype(str).map(business_group_map)
    )
    shift_plan = shift_plan.copy()
    shift_plan["location_name"] = (
        shift_plan["location_id"].astype(str).map(location_name_map)
    )
    shift_plan["business_group"] = (
        shift_plan["location_id"].astype(str).map(business_group_map)
    )

    daily_forecast = (
        forecast_df.groupby(["business_group", "date"], as_index=False)
        .agg(
            predicted_orders=("orders_prediction", "sum"),
            forecast_segment_slots=("total_slots_needed", "sum"),
            locations=("location_id", "nunique"),
        )
    )
    daily_shift = (
        shift_plan.pivot_table(
            index=["business_group", "date"],
            columns="vehicle_type",
            values="slots_to_create",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(
            columns={
                cm.AUTO: "auto_long_shift_slots",
                cm.BIKE: "bike_long_shift_slots",
            }
        )
    )
    for col in ["auto_long_shift_slots", "bike_long_shift_slots"]:
        if col not in daily_shift.columns:
            daily_shift[col] = 0
    daily_overcoverage = (
        shift_plan.groupby(["business_group", "date"], as_index=False)
        .agg(overcoverage_hours=("overcoverage_hours", "sum"))
    )
    daily_summary = (
        daily_forecast
        .merge(daily_shift, on=["business_group", "date"], how="left")
        .merge(daily_overcoverage, on=["business_group", "date"], how="left")
    )
    daily_summary["total_long_shift_slots"] = (
        daily_summary["auto_long_shift_slots"]
        + daily_summary["bike_long_shift_slots"]
    )
    daily_summary = daily_summary[
        [
            "business_group",
            "date",
            "predicted_orders",
            "forecast_segment_slots",
            "auto_long_shift_slots",
            "bike_long_shift_slots",
            "total_long_shift_slots",
            "locations",
            "overcoverage_hours",
        ]
    ]

    daily_by_vehicle = (
        shift_plan.groupby(["business_group", "date", "vehicle_type"], as_index=False)
        .agg(
            slots_to_create=("slots_to_create", "sum"),
            overcoverage_hours=("overcoverage_hours", "sum"),
        )
        .sort_values(["date", "vehicle_type"])
    )

    location_week_summary = (
        shift_plan.groupby(
            ["business_group", "location_id", "segment", "vehicle_type"],
            as_index=False,
        )
        .agg(
            slots_to_create=("slots_to_create", "sum"),
            overcoverage_hours=("overcoverage_hours", "sum"),
        )
        .sort_values(["slots_to_create"], ascending=False)
    )
    location_orders = (
        forecast_df.groupby(
            ["business_group", "location_id", "location_name", "segment"],
            as_index=False,
        )
        .agg(predicted_orders=("orders_prediction", "sum"))
    )
    location_week_summary = location_week_summary.merge(
        location_orders,
        on=["business_group", "location_id", "segment"],
        how="left",
    )
    location_week_summary = location_week_summary.sort_values(
        ["slots_to_create", "predicted_orders"],
        ascending=False,
    )

    business_simple = (
        shift_plan[
            [
                "location_id",
                "location_name",
                "business_group",
                "date",
                "segment",
                "vehicle_type",
                "shift_start",
                "shift_finish",
                "shift_hours",
                "open_intervals",
                "slots_to_create",
            ]
        ]
        .sort_values(["date", "location_id", "vehicle_type", "shift_start"])
    )

    if {
        "absorbed_partner_auto_slots",
        "absorbed_partner_bike_slots",
        "absorbed_partners",
    }.issubset(forecast_df.columns):
        absorb_by_day = (
            forecast_df.assign(
                date=pd.to_datetime(forecast_df["segment_datetime"]).dt.date.astype(str)
            )
            .groupby(["location_id", "date"], as_index=False)
            .agg(
                absorbed_partner_auto_slots=("absorbed_partner_auto_slots", "sum"),
                absorbed_partner_bike_slots=("absorbed_partner_bike_slots", "sum"),
                absorbed_partners=(
                    "absorbed_partners",
                    lambda s: ",".join(
                        dict.fromkeys(
                            name for cell in s for name in str(cell).split(",") if name
                        )
                    ),
                ),
            )
        )
        business_simple = business_simple.merge(
            absorb_by_day,
            on=["location_id", "date"],
            how="left",
        )
        business_simple["absorbed_partner_auto_slots"] = (
            business_simple["absorbed_partner_auto_slots"].fillna(0).astype(int)
        )
        business_simple["absorbed_partner_bike_slots"] = (
            business_simple["absorbed_partner_bike_slots"].fillna(0).astype(int)
        )
        business_simple["absorbed_partners"] = business_simple[
            "absorbed_partners"
        ].fillna("")

    weekly_forecast = (
        forecast_df.groupby("business_group", as_index=False)
        .agg(
            forecast_rows=("location_id", "size"),
            locations=("location_id", "nunique"),
            predicted_orders=("orders_prediction", "sum"),
            forecast_segment_slots=("total_slots_needed", "sum"),
        )
    )
    weekly_shift_total = (
        shift_plan.groupby("business_group", as_index=False)
        .agg(
            business_long_shift_slots=("slots_to_create", "sum"),
            overcoverage_hours=("overcoverage_hours", "sum"),
        )
    )
    weekly_shift_vehicle = (
        shift_plan.pivot_table(
            index="business_group",
            columns="vehicle_type",
            values="slots_to_create",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(
            columns={
                cm.AUTO: "auto_long_shift_slots",
                cm.BIKE: "bike_long_shift_slots",
            }
        )
    )
    for col in ["auto_long_shift_slots", "bike_long_shift_slots"]:
        if col not in weekly_shift_vehicle.columns:
            weekly_shift_vehicle[col] = 0
    weekly_forecast_summary = (
        weekly_forecast
        .merge(weekly_shift_total, on="business_group", how="left")
        .merge(weekly_shift_vehicle, on="business_group", how="left")
    )

    return {
        "business_shift_plan": shift_plan,
        "business_shift_plan_simple": business_simple,
        "daily_business_summary": daily_summary,
        "daily_business_summary_by_vehicle": daily_by_vehicle,
        "location_week_summary": location_week_summary,
        "work_window_report": work_window_report,
        "weekly_business_summary": weekly_forecast_summary,
    }


def save_reports(reports, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in reports.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)


def save_business_group_reports(reports, output_dir):
    group_column_by_report = {
        "business_shift_plan": "business_group",
        "business_shift_plan_simple": "business_group",
        "daily_business_summary": "business_group",
        "daily_business_summary_by_vehicle": "business_group",
        "location_week_summary": "business_group",
        "weekly_business_summary": "business_group",
    }
    all_groups = sorted(
        set(reports["weekly_business_summary"]["business_group"].dropna().astype(str))
    )
    for business_group in all_groups:
        group_dir = output_dir / business_group
        group_dir.mkdir(parents=True, exist_ok=True)
        for name, df in reports.items():
            group_col = group_column_by_report.get(name)
            if group_col is None or group_col not in df.columns:
                continue
            group_df = df[df[group_col].astype(str) == business_group].copy()
            if not group_df.empty:
                group_df.to_csv(group_dir / f"{name}.csv", index=False)


def main():
    forecast_df = load_forecast(FORECAST_PATH)
    locations_df = load_locations_metadata()
    group_to_members = build_group_to_members()
    partner_map = load_partner_donor_map()
    forecast_df, absorb_audit = absorb_partner_demand(
        forecast_df,
        partner_map,
        group_to_members,
    )
    transport_by_location = build_location_transport_lookup(locations_df)
    forecast_df = cm.enforce_transport_on_forecast(
        forecast_df,
        transport_by_location,
        group_to_members,
    )
    location_name_map = build_location_name_map(forecast_df, locations_df)
    # Enrich names for partners that were dropped from forecast.
    for row in partner_map.itertuples(index=False):
        location_name_map.setdefault(str(row.partner_location_id), str(row.partner_name))
        if str(row.kfm_donor_id):
            location_name_map.setdefault(
                str(row.kfm_donor_id),
                str(row.kfm_donor_name) or str(row.kfm_donor_id),
            )
    business_group_map = build_business_group_map(forecast_df, locations_df)
    work_hours_df = load_work_hours()
    work_interval_lookup = build_work_interval_lookup(work_hours_df)
    work_window_report = build_work_window_report(
        forecast_df,
        work_interval_lookup,
        group_to_members,
    )
    hourly_df = expand_to_hourly_demand(forecast_df)
    shift_plan = build_business_shift_plan(
        hourly_df,
        work_interval_lookup,
        group_to_members,
    )
    if shift_plan.empty:
        raise RuntimeError("No business shift rows were generated.")

    reports = build_reports(
        forecast_df,
        shift_plan,
        location_name_map,
        business_group_map,
        work_window_report,
    )
    reports["partner_absorb_audit"] = absorb_audit
    reports["partner_absorb_summary"] = build_partner_absorb_summary(
        absorb_audit,
        location_name_map,
    )
    save_reports(reports, OUTPUT_DIR)
    save_business_group_reports(reports, OUTPUT_DIR)

    print(f"Saved business reports to: {OUTPUT_DIR}")
    print(reports["weekly_business_summary"].to_string(index=False))
    print("\nPartner absorb summary:")
    if reports["partner_absorb_summary"].empty:
        print("(no partner slots absorbed)")
    else:
        print(reports["partner_absorb_summary"].to_string(index=False))
    print("\nDaily summary:")
    print(reports["daily_business_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
