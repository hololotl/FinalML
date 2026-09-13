import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

try:
    from . import main as cm
    from .partner_grouping import (
        PARTNER_DONOR_MAP_PATH,
        build_grouping_config_audit,
        load_partner_grouping,
    )
    from .orders_payment_report import (
        aggregate_payment_rates,
        build_payment_quality_audit,
        load_orders_payment_report,
    )
except ImportError:
    import main as cm
    from partner_grouping import (
        PARTNER_DONOR_MAP_PATH,
        build_grouping_config_audit,
        load_partner_grouping,
    )
    from orders_payment_report import (
        aggregate_payment_rates,
        build_payment_quality_audit,
        load_orders_payment_report,
    )


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
PAYMENT_REPORT_PATH = Path(
    os.getenv(
        "ORDERS_PAYMENT_REPORT_PATH",
        SCRIPT_DIR / "orders_payment_report.csv",
    )
)
DELIVERY_DURATION_HOURLY_PATH = Path(
    os.getenv(
        "DELIVERY_DURATION_HOURLY_PATH",
        SCRIPT_DIR
        / "res"
        / "delivery_duration"
        / "delivery_duration_by_planning_hour.csv",
    )
)
MIN_SHIFT_RUB_PER_HOUR = float(
    os.getenv("MIN_SHIFT_RUB_PER_HOUR", "0")
)
SHIFT_OVERCOVERAGE_PENALTY = float(
    os.getenv("SHIFT_OVERCOVERAGE_PENALTY", "2.0")
)
MINIMIZE_SHIFT_OVERCOVERAGE = (
    os.getenv("MINIMIZE_SHIFT_OVERCOVERAGE", "1") == "1"
)
CALIBRATION_VEHICLE_WORSEN_TOLERANCE = float(
    os.getenv("CALIBRATION_VEHICLE_WORSEN_TOLERANCE", "0.05")
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

BUSINESS_SHIFT_PLAN_SIMPLE_RU_COLUMNS = {
    "location_id": "ID ресторана",
    "location_name": "Название ресторана",
    "business_group": "Группа",
    "date": "Дата",
    "segment": "Сегмент",
    "vehicle_type": "Тип транспорта",
    "shift_start": "Начало смены",
    "shift_finish": "Конец смены",
    "shift_hours": "Длительность смены, ч",
    "open_intervals": "Часы работы ресторана",
    "slots_to_create": "Слотов создать",
    "predicted_orders_per_shift": "Заказов за смену, прогноз",
    "avg_payment_per_order_rub": "Средняя оплата за заказ, ₽",
    "p70_delivery_minutes_per_order": "P70 доставки на заказ, мин",
    "peak_orders_per_hour": "Пик заказов в час",
    "peak_hour": "Час пика",
    "max_couriers_needed_in_shift": "Макс. курьеров в смене",
    "courier_need_source": "Источник расчёта потребности",
    "predicted_shift_earnings_rub": "Прогноз заработка за смену, ₽",
    "predicted_rub_per_hour": "Прогноз, ₽/час",
    "low_expected_income": "Низкий ожидаемый доход",
    "absorbed_partner_orders": "Поглощённые заказы партнёров",
    "absorbed_partners": "Поглощённые партнёры",
}


def export_business_shift_plan_simple(df):
    return df.rename(columns=BUSINESS_SHIFT_PLAN_SIMPLE_RU_COLUMNS)


def load_forecast(path):
    df = pd.read_csv(path)
    df["location_id"] = df["location_id"].astype(str)
    df["segment_datetime"] = pd.to_datetime(
        df["segment_datetime"],
        errors="coerce",
    )
    df = df.dropna(subset=["segment_datetime"])
    return df


def load_partner_grouping_order_audit(forecast_path):
    audit_path = Path(forecast_path).parent / "partner_grouping_order_audit.csv"
    if not audit_path.exists():
        return pd.DataFrame(
            columns=[
                "source_location_id",
                "partner_name",
                "kfm_donor_id",
                "segment_datetime",
                "time_segment",
                "orders_moved",
                "action",
            ]
        )
    return pd.read_csv(audit_path)


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
    return {
        location_id: location_names.get(location_id, location_id)
        for location_id in forecast_df["location_id"].astype(str).unique()
    }


def build_business_group_map(forecast_df, locations_df):
    organization_by_location = dict(
        zip(locations_df["location_id"], locations_df["organization_id"])
    )
    return {
        location_id: (
            "kfm"
            if organization_by_location.get(location_id) in KFM_ORGANIZATION_IDS
            else "non_kfm"
        )
        for location_id in forecast_df["location_id"].astype(str).unique()
    }


def build_group_to_members():
    return load_partner_grouping().planning_to_members


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
                "orders_moved",
            ]
        )
    audit_df = audit_df.copy()
    if "partner_ids" not in audit_df.columns:
        audit_df["partner_ids"] = audit_df["source_location_id"].astype(str)
    if "partner_names" not in audit_df.columns:
        audit_df["partner_names"] = audit_df["partner_name"].astype(str)
    if "orders_moved" not in audit_df.columns:
        audit_df["orders_moved"] = 0.0
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
            orders_moved=("orders_moved", "sum"),
        )
    )
    summary["kfm_donor_name"] = summary["kfm_donor_id"].map(
        lambda x: location_name_map.get(str(x), str(x))
    )
    return summary.sort_values("orders_moved", ascending=False)


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


def build_payment_lookups(payment_rates):
    if payment_rates is None or payment_rates.empty:
        return {}, {}, (np.nan, 0.0)
    rates = payment_rates.copy()
    rates["planning_location_id"] = rates["planning_location_id"].astype(str)
    direct = {
        (row.planning_location_id, int(row.hour_from)): (
            float(row.avg_c_rate_total_rub),
            float(row.orders_count),
        )
        for row in rates.itertuples(index=False)
        if pd.notna(row.avg_c_rate_total_rub)
    }
    hourly = (
        rates.groupby("hour_from", as_index=False)
        .agg(
            orders_count=("orders_count", "sum"),
            payment_sum=("sum_c_rate_total_rub", "sum"),
        )
    )
    hourly["avg_rate"] = (
        hourly["payment_sum"] / hourly["orders_count"].replace(0, np.nan)
    )
    hourly_lookup = {
        int(row.hour_from): (float(row.avg_rate), float(row.orders_count))
        for row in hourly.itertuples(index=False)
        if pd.notna(row.avg_rate)
    }
    total_orders = float(rates["orders_count"].sum())
    global_rate = (
        float(rates["sum_c_rate_total_rub"].sum() / total_orders)
        if total_orders > 0
        else np.nan
    )
    return direct, hourly_lookup, (global_rate, total_orders)


def load_delivery_p70_rates(path=DELIVERY_DURATION_HOURLY_PATH):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    rates = pd.read_csv(path, dtype={"planning_location_id": str})
    required = {
        "planning_location_id",
        "delivery_hour",
        "vehicle_type",
        "orders_count",
        "p70_minutes",
    }
    missing = sorted(required - set(rates.columns))
    if missing:
        raise ValueError(f"Delivery duration report is missing columns: {missing}")
    return rates


def build_delivery_p70_lookups(rates):
    if rates is None or rates.empty:
        return {}, {}, {}
    rows = rates.copy()
    rows["planning_location_id"] = rows["planning_location_id"].astype(str)
    rows["orders_count"] = pd.to_numeric(rows["orders_count"], errors="coerce")
    rows["p70_minutes"] = pd.to_numeric(rows["p70_minutes"], errors="coerce")
    rows = rows.dropna(subset=["orders_count", "p70_minutes"])
    direct = {
        (
            row.planning_location_id,
            int(row.delivery_hour),
            row.vehicle_type,
        ): float(row.p70_minutes)
        for row in rows.itertuples(index=False)
    }
    rows["_weighted_p70"] = rows["p70_minutes"] * rows["orders_count"]
    hourly = (
        rows.groupby(["delivery_hour", "vehicle_type"], as_index=False)
        .agg(
            weighted_p70=("_weighted_p70", "sum"),
            orders_count=("orders_count", "sum"),
        )
    )
    hourly["p70_minutes"] = (
        hourly["weighted_p70"] / hourly["orders_count"].replace(0, np.nan)
    )
    hourly_lookup = {
        (int(row.delivery_hour), row.vehicle_type): float(row.p70_minutes)
        for row in hourly.itertuples(index=False)
    }
    global_vehicle = (
        rows.groupby("vehicle_type", as_index=False)
        .agg(
            weighted_p70=("_weighted_p70", "sum"),
            orders_count=("orders_count", "sum"),
        )
    )
    global_vehicle["p70_minutes"] = (
        global_vehicle["weighted_p70"]
        / global_vehicle["orders_count"].replace(0, np.nan)
    )
    global_lookup = {
        row.vehicle_type: float(row.p70_minutes)
        for row in global_vehicle.itertuples(index=False)
    }
    return direct, hourly_lookup, global_lookup


def lookup_delivery_p70(location_id, hour, vehicle_type, lookups):
    direct, hourly, global_vehicle = lookups
    key = (str(location_id), int(hour), vehicle_type)
    if key in direct:
        return direct[key], "location_hour"
    hour_key = (int(hour), vehicle_type)
    if hour_key in hourly:
        return hourly[hour_key], "global_hour"
    if vehicle_type in global_vehicle:
        return global_vehicle[vehicle_type], "global_vehicle"
    return np.nan, "missing"


def build_hourly_payment_quality(hourly_df):
    if hourly_df.empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(
        [
            {
                "metric": "forecast_hour_rows",
                "value": len(hourly_df),
            },
            {
                "metric": "missing_payment_rate_rows",
                "value": int(
                    hourly_df["avg_payment_per_order_rub"].isna().sum()
                ),
            },
            {
                "metric": "global_rate_fallback_rows",
                "value": int(
                    hourly_df["payment_rate_source"]
                    .astype(str)
                    .str.contains("global")
                    .sum()
                ),
            },
            {
                "metric": "uniform_hourly_profile_rows",
                "value": int(
                    hourly_df["hourly_profile_source"]
                    .astype(str)
                    .str.contains("uniform")
                    .sum()
                ),
            },
        ]
    )


def _hourly_payment_profile(location_id, hours, payment_lookups):
    direct, hourly_lookup, global_value = payment_lookups
    location_id = str(location_id)
    direct_counts = np.array(
        [direct.get((location_id, hour), (np.nan, 0.0))[1] for hour in hours],
        dtype=float,
    )
    if direct_counts.sum() > 0:
        weights = direct_counts / direct_counts.sum()
        profile_source = "location_hour_orders"
    else:
        global_counts = np.array(
            [hourly_lookup.get(hour, (np.nan, 0.0))[1] for hour in hours],
            dtype=float,
        )
        if global_counts.sum() > 0:
            weights = global_counts / global_counts.sum()
            profile_source = "global_hour_orders"
        else:
            weights = np.full(len(hours), 1.0 / len(hours))
            profile_source = "uniform"

    rates = []
    rate_sources = []
    for hour in hours:
        direct_value = direct.get((location_id, hour))
        if direct_value is not None and np.isfinite(direct_value[0]):
            rates.append(direct_value[0])
            rate_sources.append("location_hour")
        elif hour in hourly_lookup and np.isfinite(hourly_lookup[hour][0]):
            rates.append(hourly_lookup[hour][0])
            rate_sources.append("global_hour")
        else:
            rates.append(global_value[0])
            rate_sources.append("global")
    return weights, rates, rate_sources, profile_source


def expand_to_hourly_demand(
    forecast_df,
    payment_rates=None,
    delivery_p70_rates=None,
):
    payment_lookups = build_payment_lookups(payment_rates)
    delivery_p70_lookups = build_delivery_p70_lookups(delivery_p70_rates)
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
        hour_values = [
            start_hour + hour_offset
            for hour_offset in range(hours)
            if 0 <= start_hour + hour_offset < 24
        ]
        if not hour_values:
            continue
        weights, rates, rate_sources, profile_source = _hourly_payment_profile(
            row.location_id,
            hour_values,
            payment_lookups,
        )
        for hour, weight, avg_rate, rate_source in zip(
            hour_values,
            weights,
            rates,
            rate_sources,
        ):
            auto_orders = float(row.auto_order_prediction) * float(weight)
            bike_orders = float(row.bike_order_prediction) * float(weight)
            auto_p70, auto_p70_source = lookup_delivery_p70(
                row.location_id,
                hour,
                cm.AUTO,
                delivery_p70_lookups,
            )
            bike_p70, bike_p70_source = lookup_delivery_p70(
                row.location_id,
                hour,
                cm.BIKE,
                delivery_p70_lookups,
            )
            rows.append({
                "location_id": str(row.location_id),
                "date": business_date,
                "hour": hour,
                "segment": row.segment,
                "time_segment": row.time_segment,
                "vehicle_type": cm.AUTO,
                "slots_needed": int(row.auto_slots_needed),
                "predicted_orders": auto_orders,
                "avg_payment_per_order_rub": avg_rate,
                "predicted_earning_pool_rub": (
                    auto_orders * avg_rate if np.isfinite(avg_rate) else np.nan
                ),
                "payment_rate_source": rate_source,
                "hourly_profile_source": profile_source,
                "p70_delivery_minutes": auto_p70,
                "p70_delivery_source": auto_p70_source,
                "segment_predicted_orders": float(row.auto_order_prediction),
                "history_raw_slots": float(
                    getattr(row, "auto_history_raw_slots", np.nan)
                ),
                "cycle_time_raw_slots": float(
                    getattr(row, "auto_cycle_time_raw_slots", np.nan)
                ),
                "selected_raw_slots": float(
                    getattr(row, "auto_selected_raw_slots", np.nan)
                ),
                "slot_demand_source": str(
                    getattr(row, "auto_slot_demand_source", "")
                ),
                "orders_per_slot": float(
                    getattr(row, "auto_orders_per_slot", np.nan)
                ),
                "slot_capacity_source": str(
                    getattr(row, "auto_slot_capacity_source", "")
                ),
                "p80_delivery_minutes": float(
                    getattr(row, "auto_delivery_percentile_minutes", np.nan)
                ),
                "delivery_duration_source": str(
                    getattr(row, "auto_delivery_duration_source", "")
                ),
                "safety_buffer": float(
                    getattr(row, "safety_buffer", cm.SAFETY_BUFFER)
                ),
            })
            rows.append({
                "location_id": str(row.location_id),
                "date": business_date,
                "hour": hour,
                "segment": row.segment,
                "time_segment": row.time_segment,
                "vehicle_type": cm.BIKE,
                "slots_needed": int(row.bike_slots_needed),
                "predicted_orders": bike_orders,
                "avg_payment_per_order_rub": avg_rate,
                "predicted_earning_pool_rub": (
                    bike_orders * avg_rate if np.isfinite(avg_rate) else np.nan
                ),
                "payment_rate_source": rate_source,
                "hourly_profile_source": profile_source,
                "p70_delivery_minutes": bike_p70,
                "p70_delivery_source": bike_p70_source,
                "segment_predicted_orders": float(row.bike_order_prediction),
                "history_raw_slots": float(
                    getattr(row, "bike_history_raw_slots", np.nan)
                ),
                "cycle_time_raw_slots": float(
                    getattr(row, "bike_cycle_time_raw_slots", np.nan)
                ),
                "selected_raw_slots": float(
                    getattr(row, "bike_selected_raw_slots", np.nan)
                ),
                "slot_demand_source": str(
                    getattr(row, "bike_slot_demand_source", "")
                ),
                "orders_per_slot": float(
                    getattr(row, "bike_orders_per_slot", np.nan)
                ),
                "slot_capacity_source": str(
                    getattr(row, "bike_slot_capacity_source", "")
                ),
                "p80_delivery_minutes": float(
                    getattr(row, "bike_delivery_percentile_minutes", np.nan)
                ),
                "delivery_duration_source": str(
                    getattr(row, "bike_delivery_duration_source", "")
                ),
                "safety_buffer": float(
                    getattr(row, "safety_buffer", cm.SAFETY_BUFFER)
                ),
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
            predicted_earning_pool_rub=("predicted_earning_pool_rub", "sum"),
            avg_payment_per_order_rub=("avg_payment_per_order_rub", "mean"),
            payment_rate_source=(
                "payment_rate_source",
                lambda values: ",".join(dict.fromkeys(values)),
            ),
            hourly_profile_source=(
                "hourly_profile_source",
                lambda values: ",".join(dict.fromkeys(values)),
            ),
            p70_delivery_minutes=("p70_delivery_minutes", "mean"),
            p70_delivery_source=(
                "p70_delivery_source",
                lambda values: ",".join(dict.fromkeys(values)),
            ),
            time_segment=("time_segment", "first"),
            segment_predicted_orders=("segment_predicted_orders", "sum"),
            history_raw_slots=("history_raw_slots", "max"),
            cycle_time_raw_slots=("cycle_time_raw_slots", "max"),
            selected_raw_slots=("selected_raw_slots", "max"),
            slot_demand_source=("slot_demand_source", "first"),
            orders_per_slot=("orders_per_slot", "max"),
            slot_capacity_source=("slot_capacity_source", "first"),
            p80_delivery_minutes=("p80_delivery_minutes", "max"),
            delivery_duration_source=("delivery_duration_source", "first"),
            safety_buffer=("safety_buffer", "max"),
        )
    )


def build_hour_arrays(group):
    demand = np.zeros(24, dtype=int)
    orders = np.zeros(24, dtype=float)
    earnings = np.zeros(24, dtype=float)
    for row in group.itertuples(index=False):
        demand[int(row.hour)] = max(demand[int(row.hour)], int(row.slots_needed))
        orders[int(row.hour)] += float(row.predicted_orders)
        value = getattr(row, "predicted_earning_pool_rub", 0.0)
        earnings[int(row.hour)] += float(value) if pd.notna(value) else 0.0
    return demand, orders, earnings


def build_hourly_courier_need_explanation(
    hourly_df,
    location_name_map,
    business_group_map,
    work_interval_lookup,
    group_to_members,
):
    if hourly_df.empty:
        return pd.DataFrame()
    report = hourly_df[
        (hourly_df["slots_needed"] > 0)
        | (hourly_df["predicted_orders"] > 0)
    ].copy()
    first_date = pd.to_datetime(report["date"]).min().date()
    report = report[
        pd.to_datetime(report["date"]).dt.date == first_date
    ].copy()
    interval_cache = {}
    for location_id, date in report[["location_id", "date"]].drop_duplicates().itertuples(
        index=False
    ):
        interval_cache[(str(location_id), str(date))] = get_open_intervals(
            location_id,
            pd.Timestamp(date).dayofweek,
            work_interval_lookup,
            group_to_members,
        )
    inside_work_hours = [
        hour_is_inside_open_intervals(
            int(row.hour),
            interval_cache.get((str(row.location_id), str(row.date)), []),
        )
        for row in report.itertuples(index=False)
    ]
    report = report.loc[inside_work_hours].copy()
    report["location_name"] = (
        report["location_id"].astype(str).map(location_name_map)
    )
    report["business_group"] = (
        report["location_id"].astype(str).map(business_group_map)
    )
    report["hour_window"] = report["hour"].map(
        lambda hour: f"{int(hour):02d}:00-{int(hour) + 1:02d}:00"
    )
    report["calculation_window_hours"] = report["time_segment"].map(
        cm.segment_hours
    )
    report["historical_capacity_orders_per_hour"] = (
        report["orders_per_slot"]
        / report["calculation_window_hours"].replace(0, np.nan)
    )
    report["p80_round_trip_minutes"] = (
        report["p80_delivery_minutes"] * cm.CYCLE_TIME_RETURN_MULTIPLIER
    )

    source_labels = {
        "history": "историческая производительность",
        "cycle_time": "время полного цикла доставки",
    }

    def explain(row):
        source = source_labels.get(
            str(row.slot_demand_source),
            str(row.slot_demand_source) or "fallback",
        )
        return (
            f"Нужно {int(row.slots_needed)} кур.; "
            f"прогноз {row.predicted_orders:.1f} заказа/ч; "
            f"окно {row.time_segment}: "
            f"история {row.history_raw_slots:.2f}, "
            f"цикл {row.cycle_time_raw_slots:.2f}; "
            f"выбрано: {source}; "
            f"страховой запас {max(row.safety_buffer - 1, 0):.0%}"
        )

    report["courier_need_reason"] = report.apply(explain, axis=1)
    columns = [
        "location_id",
        "location_name",
        "business_group",
        "date",
        "hour",
        "hour_window",
        "segment",
        "time_segment",
        "vehicle_type",
        "predicted_orders",
        "segment_predicted_orders",
        "p70_delivery_minutes",
        "p80_delivery_minutes",
        "p80_round_trip_minutes",
        "historical_capacity_orders_per_hour",
        "history_raw_slots",
        "cycle_time_raw_slots",
        "selected_raw_slots",
        "safety_buffer",
        "slots_needed",
        "slot_demand_source",
        "slot_capacity_source",
        "delivery_duration_source",
        "payment_rate_source",
        "p70_delivery_source",
        "courier_need_reason",
    ]
    numeric_columns = [
        "predicted_orders",
        "segment_predicted_orders",
        "p70_delivery_minutes",
        "p80_delivery_minutes",
        "p80_round_trip_minutes",
        "historical_capacity_orders_per_hour",
        "history_raw_slots",
        "cycle_time_raw_slots",
        "selected_raw_slots",
        "safety_buffer",
    ]
    report[numeric_columns] = report[numeric_columns].round(2)
    return report[columns].sort_values(
        ["date", "location_id", "vehicle_type", "hour"]
    )


def build_single_courier_open_shifts(
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    hourly_earnings,
    open_intervals,
):
    rows = []
    candidate_intervals = open_intervals or [(0.0, 24.0)]
    for interval_start, interval_finish in candidate_intervals:
        start_bound = int(np.ceil(interval_start))
        finish_bound = int(np.floor(interval_finish))
        if finish_bound <= start_bound:
            continue
        if MINIMIZE_SHIFT_OVERCOVERAGE:
            interval_active = np.where(demand[start_bound:finish_bound] > 0)[0]
            if len(interval_active) == 0:
                continue
            demand_start = start_bound + int(interval_active.min())
            demand_finish = start_bound + int(interval_active.max()) + 1
        else:
            demand_start = start_bound
            demand_finish = finish_bound

        hourly_income = np.zeros(24, dtype=float)
        active = demand > 0
        hourly_income[active] = (
            hourly_earnings[active] / np.maximum(demand[active], 1)
        )
        chunks = split_shift_into_max_hours(
            demand_start,
            demand_finish,
            MAX_SHIFT_HOURS,
            hourly_income,
        )
        for chunk_start, chunk_finish in chunks:
            active_hours = int(
                (demand[chunk_start:chunk_finish] > 0).sum()
            )
            duration_hours = chunk_finish - chunk_start
            rows.append({
                "location_id": location_id,
                "date": date,
                "segment": segment,
                "vehicle_type": vehicle_type,
                "shift_template": (
                    "single_layer_demand_span"
                    if len(chunks) == 1
                    else "single_layer_demand_span_earnings_split"
                ),
                "shift_start": f"{chunk_start:02d}:00",
                "shift_finish": f"{chunk_finish:02d}:00",
                "shift_hours": duration_hours,
                "open_intervals": format_intervals(candidate_intervals),
                "slots_to_create": 1,
                "covered_need_hours": active_hours,
                "template_hours": duration_hours,
                "overcoverage_hours": duration_hours - active_hours,
                "covered_predicted_orders": float(
                    hourly_orders[chunk_start:chunk_finish].sum()
                ),
                **_shift_income_metrics(
                    demand,
                    hourly_earnings,
                    hourly_orders,
                    chunk_start,
                    chunk_finish,
                    1,
                ),
            })
    return rows


def choose_shift_template(residual, open_intervals, hourly_income=None):
    candidates = []
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
        earnings = (
            float(np.asarray(hourly_income)[start_hour:end_hour].sum())
            if hourly_income is not None
            else 0.0
        )
        candidates.append(
            (
                covered,
                over_hours,
                earnings,
                template_name,
                start_hour,
                end_hour,
                duration_hours,
            )
        )
    if not candidates:
        return None
    target_earnings = float(np.median([candidate[2] for candidate in candidates]))
    if MINIMIZE_SHIFT_OVERCOVERAGE:
        score = lambda candidate: (
            candidate[0] - SHIFT_OVERCOVERAGE_PENALTY * candidate[1],
            -candidate[1],
            candidate[0],
            -abs(candidate[2] - target_earnings),
            -candidate[6],
        )
    else:
        score = lambda candidate: (
            candidate[0],
            -abs(candidate[2] - target_earnings),
            -candidate[1],
            -candidate[6],
        )
    best = max(candidates, key=score)
    _, _, _, template_name, start_hour, end_hour, duration_hours = best
    return template_name, start_hour, end_hour, duration_hours


def choose_fallback_shift(residual, open_intervals, hourly_income=None):
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


def split_shift_into_max_hours(
    shift_start,
    shift_finish,
    max_hours=MAX_SHIFT_HOURS,
    hour_values=None,
):
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
    if chunk_count == 2 and hour_values is not None:
        values = np.asarray(hour_values, dtype=float)
        candidates = []
        min_boundary = max(start + 1, finish - max_hours)
        max_boundary = min(finish - 1, start + max_hours)
        for boundary in range(min_boundary, max_boundary + 1):
            left = float(values[start:boundary].sum())
            right = float(values[boundary:finish].sum())
            earning_gap = abs(left - right)
            duration_gap = abs((boundary - start) - (finish - boundary))
            candidates.append((earning_gap, duration_gap, boundary))
        if candidates:
            boundary = min(candidates)[2]
            return [(start, boundary), (boundary, finish)]
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


def _shift_income_metrics(
    demand,
    hourly_earnings,
    hourly_orders,
    shift_start,
    shift_finish,
    layer,
):
    duration_hours = max(int(shift_finish) - int(shift_start), 0)
    expected_earnings = 0.0
    expected_orders = 0.0
    for hour in range(int(shift_start), int(shift_finish)):
        if demand[hour] >= layer and demand[hour] > 0:
            expected_earnings += float(hourly_earnings[hour]) / float(demand[hour])
            expected_orders += float(hourly_orders[hour]) / float(demand[hour])
    rub_per_hour = (
        expected_earnings / duration_hours if duration_hours > 0 else 0.0
    )
    return {
        "predicted_shift_earnings_rub": expected_earnings,
        "assigned_predicted_orders": expected_orders,
        "predicted_rub_per_hour": rub_per_hour,
        "low_expected_income": bool(
            MIN_SHIFT_RUB_PER_HOUR > 0
            and rub_per_hour < MIN_SHIFT_RUB_PER_HOUR
        ),
    }


def build_shift_plan_for_group(
    location_id,
    date,
    segment,
    vehicle_type,
    group,
    open_intervals,
):
    residual, hourly_orders, hourly_earnings = build_hour_arrays(group)
    for hour in range(24):
        if not hour_is_inside_open_intervals(hour, open_intervals):
            residual[hour] = 0
            hourly_orders[hour] = 0
            hourly_earnings[hour] = 0

    if int(residual.max()) == 1:
        return build_single_courier_open_shifts(
            location_id,
            date,
            segment,
            vehicle_type,
            residual,
            hourly_orders,
            hourly_earnings,
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
            hourly_earnings,
            open_intervals,
        )

    rows = []

    while residual.max() > 0:
        hourly_income = hourly_earnings / np.maximum(residual, 1)
        chosen = choose_shift_template(
            residual,
            open_intervals,
            hourly_income,
        )
        if chosen is None:
            chosen = choose_fallback_shift(
                residual,
                open_intervals,
                hourly_income,
            )
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
            **_shift_income_metrics(
                residual + (active_hours.astype(int)),
                hourly_earnings,
                hourly_orders,
                start_hour,
                end_hour,
                1,
            ),
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
    hourly_earnings,
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
        layer_income = np.zeros(24, dtype=float)
        active = demand >= layer
        layer_income[active] = (
            hourly_earnings[active] / np.maximum(demand[active], 1)
        )
        chunks = split_shift_into_max_hours(
            shift_start,
            shift_finish,
            MAX_SHIFT_HOURS,
            layer_income,
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
            **_shift_income_metrics(
                demand,
                hourly_earnings,
                hourly_orders,
                chunk_start,
                chunk_finish,
                layer,
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


def _merge_adjacent_layer_runs(shift_start, shift_finish, uncovered_runs):
    """Merge only runs touching the same layer's main shift.

    A separated peak remains a separate short shift. A prefix such as 07-10
    touching a 10-21 shift becomes 07-21 and is subsequently split by the
    regular max-hours/earnings logic.
    """
    merged_start = int(shift_start)
    merged_finish = int(shift_finish)
    remaining = [(int(start), int(finish)) for start, finish in uncovered_runs]
    changed = True
    while changed:
        changed = False
        next_remaining = []
        for run_start, run_finish in remaining:
            if run_finish == merged_start:
                merged_start = run_start
                changed = True
            elif run_start == merged_finish:
                merged_finish = run_finish
                changed = True
            else:
                next_remaining.append((run_start, run_finish))
        remaining = next_remaining
    return merged_start, merged_finish, remaining


def _build_layer_rows_for_interval(
    location_id,
    date,
    segment,
    vehicle_type,
    demand,
    hourly_orders,
    hourly_earnings,
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
                if MINIMIZE_SHIFT_OVERCOVERAGE:
                    active_positions = np.where(layer_mask)[0]
                    shift_start = start_bound + int(active_positions.min())
                    shift_finish = start_bound + int(active_positions.max()) + 1
                    template_name = "first_layer_demand_span"
                else:
                    shift_start = start_bound
                    shift_finish = finish_bound
                    template_name = "full_open_window"
                enforce_max_shift_hours = True
            else:
                residual = np.zeros(24, dtype=int)
                residual[start_bound:finish_bound] = layer_mask.astype(int)
                layer_income = np.zeros(24, dtype=float)
                active = demand >= layer
                layer_income[active] = (
                    hourly_earnings[active] / np.maximum(demand[active], 1)
                )
                chosen = choose_shift_template(
                    residual,
                    [open_interval],
                    layer_income,
                )
                if chosen is None:
                    chosen = choose_fallback_shift(
                        residual,
                        [open_interval],
                        layer_income,
                    )
                if chosen is None:
                    continue
                template_name, shift_start, shift_finish, _ = chosen
                shift_finish = max(int(shift_finish), int(active_finish))
                shift_finish = min(int(shift_finish), int(finish_bound))
                enforce_max_shift_hours = True
            covered_mask = np.zeros_like(layer_mask, dtype=bool)
            covered_start = max(int(shift_start), start_bound) - start_bound
            covered_finish = min(int(shift_finish), finish_bound) - start_bound
            if covered_finish > covered_start:
                covered_mask[covered_start:covered_finish] = True
            uncovered_mask = layer_mask & ~covered_mask
            uncovered_runs = [
                (
                    start_bound + int(run_start),
                    start_bound + int(run_finish),
                )
                for run_start, run_finish in _iter_contiguous_true_runs(
                    uncovered_mask
                )
            ]
            merged_start, merged_finish, remaining_runs = (
                _merge_adjacent_layer_runs(
                    shift_start,
                    shift_finish,
                    uncovered_runs,
                )
            )
            if merged_start != int(shift_start) or merged_finish != int(shift_finish):
                template_name = f"{template_name}_with_adjacent_demand"
            _append_shift_row(
                rows,
                location_id,
                date,
                segment,
                vehicle_type,
                demand,
                hourly_orders,
                hourly_earnings,
                open_intervals_display,
                merged_start,
                merged_finish,
                layer,
                template_name,
                enforce_max_shift_hours=enforce_max_shift_hours,
            )
            for run_start, run_finish in remaining_runs:
                _append_shift_row(
                    rows,
                    location_id,
                    date,
                    segment,
                    vehicle_type,
                    demand,
                    hourly_orders,
                    hourly_earnings,
                    open_intervals_display,
                    run_start,
                    run_finish,
                    layer,
                    "demand_layer_gap",
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
                hourly_earnings,
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
    hourly_earnings,
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
                hourly_earnings,
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
    hourly_earnings,
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
                hourly_earnings,
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
            hourly_earnings,
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
    grouped = (
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
            predicted_total_earnings_rub=(
                "predicted_shift_earnings_rub",
                "sum",
            ),
            assigned_predicted_orders_total=(
                "assigned_predicted_orders",
                "sum",
            ),
            low_expected_income=("low_expected_income", "any"),
        )
        .sort_values(
            ["date", "location_id", "vehicle_type", "shift_start", "shift_finish"]
        )
    )
    grouped["predicted_shift_earnings_rub"] = (
        grouped["predicted_total_earnings_rub"]
        / grouped["slots_to_create"].clip(lower=1)
    )
    grouped["predicted_rub_per_hour"] = (
        grouped["predicted_shift_earnings_rub"]
        / grouped["shift_hours"].replace(0, np.nan)
    )
    grouped["predicted_orders_per_shift"] = (
        grouped["assigned_predicted_orders_total"]
        / grouped["slots_to_create"].clip(lower=1)
    )
    grouped["avg_payment_per_order_rub"] = (
        grouped["predicted_total_earnings_rub"]
        / grouped["assigned_predicted_orders_total"].replace(0, np.nan)
    )
    grouped["low_expected_income"] = (
        grouped["low_expected_income"]
        | (
            (MIN_SHIFT_RUB_PER_HOUR > 0)
            & (grouped["predicted_rub_per_hour"] < MIN_SHIFT_RUB_PER_HOUR)
        )
    )
    return grouped


def attach_shift_delivery_p70(shift_plan, hourly_df):
    if shift_plan.empty:
        return shift_plan
    group_columns = ["location_id", "date", "segment", "vehicle_type"]
    hourly_groups = {
        key: group.set_index("hour")
        for key, group in hourly_df.groupby(group_columns)
    }
    p70_values = []
    p70_sources = []
    peak_orders_values = []
    peak_hours = []
    max_couriers_values = []
    need_sources = []
    for row in shift_plan.itertuples(index=False):
        key = tuple(getattr(row, column) for column in group_columns)
        hourly = hourly_groups.get(key)
        start = int(str(row.shift_start).split(":")[0])
        finish = int(str(row.shift_finish).split(":")[0])
        weighted_minutes = 0.0
        order_weight = 0.0
        sources = []
        if hourly is not None:
            for hour in range(start, finish):
                if hour not in hourly.index:
                    continue
                hour_row = hourly.loc[hour]
                p70 = float(hour_row["p70_delivery_minutes"])
                if not np.isfinite(p70):
                    continue
                slots = max(float(hour_row["slots_needed"]), 1.0)
                orders_per_courier = float(hour_row["predicted_orders"]) / slots
                weighted_minutes += p70 * orders_per_courier
                order_weight += orders_per_courier
                sources.extend(
                    str(hour_row["p70_delivery_source"]).split(",")
                )
            shift_hours = hourly.loc[
                hourly.index.intersection(range(start, finish))
            ]
            if not shift_hours.empty:
                peak_index = shift_hours["predicted_orders"].idxmax()
                peak_orders_values.append(
                    float(shift_hours.loc[peak_index, "predicted_orders"])
                )
                peak_hours.append(f"{int(peak_index):02d}:00")
                max_couriers_values.append(
                    int(shift_hours["slots_needed"].max())
                )
                need_sources.append(
                    str(shift_hours.loc[peak_index, "slot_demand_source"])
                )
            else:
                peak_orders_values.append(np.nan)
                peak_hours.append("")
                max_couriers_values.append(0)
                need_sources.append("")
        else:
            peak_orders_values.append(np.nan)
            peak_hours.append("")
            max_couriers_values.append(0)
            need_sources.append("")
        p70_values.append(
            weighted_minutes / order_weight if order_weight > 0 else np.nan
        )
        p70_sources.append(",".join(dict.fromkeys(source for source in sources if source)))
    result = shift_plan.copy()
    result["p70_delivery_minutes_per_order"] = p70_values
    result["p70_delivery_source"] = p70_sources
    result["peak_orders_per_hour"] = peak_orders_values
    result["peak_hour"] = peak_hours
    result["max_couriers_needed_in_shift"] = max_couriers_values
    result["courier_need_source"] = need_sources
    return result


def build_shift_earnings_audit(shift_plan):
    if shift_plan.empty:
        return pd.DataFrame()
    group_columns = ["location_id", "date", "segment", "vehicle_type"]
    rows = []
    for key, group in shift_plan.groupby(group_columns):
        per_slot_earnings = np.repeat(
            group["predicted_shift_earnings_rub"].to_numpy(),
            group["slots_to_create"].astype(int).to_numpy(),
        )
        mean_earnings = (
            float(per_slot_earnings.mean()) if len(per_slot_earnings) else 0.0
        )
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "shifts_count": int(group["slots_to_create"].sum()),
                "expected_earnings_mean_rub": mean_earnings,
                "expected_earnings_min_rub": (
                    float(per_slot_earnings.min())
                    if len(per_slot_earnings)
                    else 0.0
                ),
                "expected_earnings_max_rub": (
                    float(per_slot_earnings.max())
                    if len(per_slot_earnings)
                    else 0.0
                ),
                "earnings_coefficient_of_variation": (
                    float(per_slot_earnings.std() / mean_earnings)
                    if mean_earnings > 0
                    else np.nan
                ),
                "rub_per_hour_p10": float(
                    group["predicted_rub_per_hour"].quantile(0.10)
                ),
                "rub_per_hour_median": float(
                    group["predicted_rub_per_hour"].median()
                ),
                "low_expected_income_shifts": int(
                    group.loc[group["low_expected_income"], "slots_to_create"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_control_week_validation(
    hourly_df,
    shift_plan,
    work_interval_lookup=None,
    group_to_members=None,
):
    if hourly_df.empty or shift_plan.empty:
        return pd.DataFrame(columns=["metric", "value"])
    start_date = pd.to_datetime(hourly_df["date"]).min()
    finish_date = start_date + pd.Timedelta(days=7)
    hourly = hourly_df[
        pd.to_datetime(hourly_df["date"]).between(
            start_date,
            finish_date,
            inclusive="left",
        )
    ].copy()
    shifts = shift_plan[
        pd.to_datetime(shift_plan["date"]).between(
            start_date,
            finish_date,
            inclusive="left",
        )
    ].copy()

    group_columns = ["location_id", "date", "segment", "vehicle_type"]
    coverage_shortfall = 0
    required_courier_hours = 0
    planned_courier_hours = 0
    max_demand = (
        hourly.groupby(group_columns)["slots_needed"].max().to_dict()
    )
    shifts_by_group = {
        key: group
        for key, group in shifts.groupby(group_columns)
    }
    for key, group in hourly.groupby(group_columns):
        location_id, date, _, _ = key
        open_intervals = get_open_intervals(
            location_id,
            pd.Timestamp(date).dayofweek,
            work_interval_lookup or {},
            group_to_members or {},
        )
        coverage = np.zeros(24, dtype=int)
        for row in shifts_by_group.get(key, pd.DataFrame()).itertuples(index=False):
            start = int(str(row.shift_start).split(":")[0])
            finish = int(str(row.shift_finish).split(":")[0])
            coverage[start:finish] += int(row.slots_to_create)
            planned_courier_hours += (
                finish - start
            ) * int(row.slots_to_create)
        for row in group.itertuples(index=False):
            required = (
                int(row.slots_needed)
                if hour_is_inside_open_intervals(
                    int(row.hour),
                    open_intervals,
                )
                else 0
            )
            required_courier_hours += required
            coverage_shortfall += max(required - coverage[int(row.hour)], 0)

    invalid_long_shifts = 0
    all_long_shifts = 0
    for row in shifts.itertuples(index=False):
        key = tuple(getattr(row, column) for column in group_columns)
        if row.shift_hours > MAX_SHIFT_HOURS:
            all_long_shifts += int(row.slots_to_create)
        if row.shift_hours > MAX_SHIFT_HOURS and max_demand.get(key, 0) > 1:
            invalid_long_shifts += int(row.slots_to_create)

    earnings = shifts["predicted_rub_per_hour"].replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(
        [
            {"metric": "control_week_start", "value": start_date.date()},
            {
                "metric": "control_week_finish",
                "value": finish_date.date(),
            },
            {
                "metric": "required_courier_hours",
                "value": required_courier_hours,
            },
            {
                "metric": "planned_courier_hours",
                "value": planned_courier_hours,
            },
            {
                "metric": "coverage_shortfall_courier_hours",
                "value": coverage_shortfall,
            },
            {
                "metric": "multi_courier_shifts_over_12h",
                "value": invalid_long_shifts,
            },
            {
                "metric": "all_shifts_over_12h",
                "value": all_long_shifts,
            },
            {
                "metric": "shift_earnings_p10_rub",
                "value": shifts["predicted_shift_earnings_rub"].quantile(0.10),
            },
            {
                "metric": "shift_earnings_median_rub",
                "value": shifts["predicted_shift_earnings_rub"].median(),
            },
            {
                "metric": "rub_per_hour_p10",
                "value": earnings.quantile(0.10),
            },
            {
                "metric": "rub_per_hour_median",
                "value": earnings.median(),
            },
            {
                "metric": "low_expected_income_shifts",
                "value": int(
                    shifts.loc[
                        shifts["low_expected_income"],
                        "slots_to_create",
                    ].sum()
                ),
            },
        ]
    )


def forecast_with_baseline_slots(forecast_df):
    required = {
        "baseline_auto_slots_needed",
        "baseline_bike_slots_needed",
        "baseline_total_slots_needed",
    }
    if not required.issubset(forecast_df.columns):
        return None
    baseline = forecast_df.copy()
    for vehicle_type in [cm.AUTO, cm.BIKE]:
        baseline[f"{vehicle_type}_slots_needed"] = baseline[
            f"baseline_{vehicle_type}_slots_needed"
        ].astype(int)
        baseline[f"{vehicle_type}_couriers_needed"] = baseline[
            f"{vehicle_type}_slots_needed"
        ]
    baseline["total_slots_needed"] = baseline[
        "baseline_total_slots_needed"
    ].astype(int)
    baseline["total_couriers_needed"] = baseline["total_slots_needed"]
    return baseline


def _plan_hours_by_vehicle(shift_plan):
    if shift_plan is None or shift_plan.empty:
        return {cm.AUTO: 0.0, cm.BIKE: 0.0}
    rows = shift_plan.copy()
    rows["courier_hours"] = (
        pd.to_numeric(rows["shift_hours"], errors="coerce").fillna(0)
        * pd.to_numeric(rows["slots_to_create"], errors="coerce").fillna(0)
    )
    totals = rows.groupby("vehicle_type")["courier_hours"].sum()
    return {
        vehicle_type: float(totals.get(vehicle_type, 0.0))
        for vehicle_type in [cm.AUTO, cm.BIKE]
    }


def load_actual_courier_hours(forecast_path):
    path = Path(forecast_path).parent / "actual_slots_by_window.csv"
    if not path.exists():
        return {cm.AUTO: 0.0, cm.BIKE: 0.0}
    actual = pd.read_csv(path)
    if actual.empty or "shift_overlap_hours" not in actual.columns:
        return {cm.AUTO: 0.0, cm.BIKE: 0.0}
    actual["shift_overlap_hours"] = pd.to_numeric(
        actual["shift_overlap_hours"],
        errors="coerce",
    ).fillna(0)
    totals = actual.groupby("vehicle_type")["shift_overlap_hours"].sum()
    return {
        vehicle_type: float(totals.get(vehicle_type, 0.0))
        for vehicle_type in [cm.AUTO, cm.BIKE]
    }


def build_courier_hour_calibration_comparison(
    baseline_plan,
    candidate_plan,
    actual_hours,
    candidate_validation,
):
    baseline_hours = _plan_hours_by_vehicle(baseline_plan)
    candidate_hours = _plan_hours_by_vehicle(candidate_plan)
    actual_total = sum(actual_hours.values())
    baseline_total = sum(baseline_hours.values())
    candidate_total = sum(candidate_hours.values())

    validation_metrics = dict(
        zip(candidate_validation["metric"], candidate_validation["value"])
    )
    constraints_pass = (
        float(validation_metrics.get("coverage_shortfall_courier_hours", 0)) == 0
        and float(validation_metrics.get("all_shifts_over_12h", 0)) == 0
    )
    if actual_total > 0:
        total_improves = abs(candidate_total - actual_total) < abs(
            baseline_total - actual_total
        )
        vehicle_pass = all(
            abs(candidate_hours[vehicle_type] - actual_hours[vehicle_type])
            <= abs(baseline_hours[vehicle_type] - actual_hours[vehicle_type])
            + CALIBRATION_VEHICLE_WORSEN_TOLERANCE
            * max(actual_hours[vehicle_type], 1.0)
            for vehicle_type in [cm.AUTO, cm.BIKE]
        )
        accepted = constraints_pass and total_improves and vehicle_pass
        reason = (
            "improved_holdout_without_coverage_loss"
            if accepted
            else "candidate_rejected_by_holdout_gate"
        )
    else:
        accepted = constraints_pass
        reason = (
            "no_actual_holdout_use_validated_candidate"
            if accepted
            else "candidate_rejected_by_constraints"
        )

    rows = []
    for model, hours in [
        ("baseline", baseline_hours),
        ("candidate", candidate_hours),
    ]:
        for vehicle_type in [cm.AUTO, cm.BIKE, "total"]:
            actual_value = (
                actual_total
                if vehicle_type == "total"
                else actual_hours[vehicle_type]
            )
            planned_value = (
                sum(hours.values())
                if vehicle_type == "total"
                else hours[vehicle_type]
            )
            rows.append(
                {
                    "model": model,
                    "vehicle_type": vehicle_type,
                    "planned_courier_hours": planned_value,
                    "actual_courier_hours": actual_value,
                    "absolute_error_hours": abs(planned_value - actual_value),
                    "courier_hour_ape": (
                        abs(planned_value - actual_value) / actual_value
                        if actual_value > 0
                        else np.nan
                    ),
                    "candidate_accepted": accepted,
                    "decision_reason": reason,
                }
            )
    return pd.DataFrame(rows), accepted


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
                "predicted_orders_per_shift",
                "avg_payment_per_order_rub",
                "p70_delivery_minutes_per_order",
                "peak_orders_per_hour",
                "peak_hour",
                "max_couriers_needed_in_shift",
                "courier_need_source",
                "predicted_shift_earnings_rub",
                "predicted_rub_per_hour",
                "low_expected_income",
            ]
        ]
        .sort_values(["date", "location_id", "vehicle_type", "shift_start"])
    )
    money_and_order_columns = [
        "predicted_orders_per_shift",
        "avg_payment_per_order_rub",
        "p70_delivery_minutes_per_order",
        "peak_orders_per_hour",
        "predicted_shift_earnings_rub",
        "predicted_rub_per_hour",
    ]
    business_simple[money_and_order_columns] = business_simple[
        money_and_order_columns
    ].round(2)

    if {
        "absorbed_partner_orders",
        "absorbed_partners",
    }.issubset(forecast_df.columns):
        absorb_by_day = (
            forecast_df.assign(
                date=pd.to_datetime(forecast_df["segment_datetime"]).dt.date.astype(str)
            )
            .groupby(["location_id", "date"], as_index=False)
            .agg(
                absorbed_partner_orders=("absorbed_partner_orders", "sum"),
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
        business_simple["absorbed_partner_orders"] = (
            business_simple["absorbed_partner_orders"].fillna(0.0)
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
        out_df = df
        if name == "business_shift_plan_simple":
            out_df = export_business_shift_plan_simple(df)
        out_df.to_csv(output_dir / f"{name}.csv", index=False)


def save_business_group_reports(reports, output_dir):
    group_column_by_report = {
        "business_shift_plan": "business_group",
        "business_shift_plan_simple": "business_group",
        "daily_business_summary": "business_group",
        "daily_business_summary_by_vehicle": "business_group",
        "location_week_summary": "business_group",
        "weekly_business_summary": "business_group",
        "hourly_courier_need_explanation": "business_group",
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
                if name == "business_shift_plan_simple":
                    group_df = export_business_shift_plan_simple(group_df)
                group_df.to_csv(group_dir / f"{name}.csv", index=False)


def main():
    global MINIMIZE_SHIFT_OVERCOVERAGE

    forecast_df = load_forecast(FORECAST_PATH)
    locations_df = load_locations_metadata()
    partner_grouping = load_partner_grouping()
    group_to_members = partner_grouping.planning_to_members
    partner_map = partner_grouping.mapping_df
    forecast_location_ids = set(forecast_df["location_id"].astype(str))
    legacy_group_ids = sorted(
        location_id
        for location_id in forecast_location_ids
        if location_id.startswith("grp_")
    )
    if legacy_group_ids:
        raise ValueError(
            "Forecast contains legacy grp_* IDs. Regenerate week_model and "
            f"courier_model with partner_donor_map.csv: {legacy_group_ids[:10]}"
        )
    absorbed_sources = set(partner_grouping.location_to_planning)
    ungrouped_sources = sorted(forecast_location_ids & absorbed_sources)
    if ungrouped_sources:
        raise ValueError(
            "Forecast still contains absorb partners. Group predictions before "
            f"slot calculation: {ungrouped_sources[:10]}"
        )
    absorb_audit = load_partner_grouping_order_audit(FORECAST_PATH)
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
    payment_rows, payment_metadata = load_orders_payment_report(
        PAYMENT_REPORT_PATH
    )
    payment_rates = aggregate_payment_rates(
        payment_rows,
        partner_grouping,
    )
    payment_quality_audit = build_payment_quality_audit(
        payment_rows,
        payment_metadata,
        forecast_df["segment_datetime"].min(),
    )
    delivery_p70_rates = load_delivery_p70_rates()
    hourly_df = expand_to_hourly_demand(
        forecast_df,
        payment_rates,
        delivery_p70_rates,
    )
    payment_quality_audit = pd.concat(
        [
            payment_quality_audit,
            build_hourly_payment_quality(hourly_df),
        ],
        ignore_index=True,
    )
    candidate_shift_plan = build_business_shift_plan(
        hourly_df,
        work_interval_lookup,
        group_to_members,
    )
    if candidate_shift_plan.empty:
        raise RuntimeError("No business shift rows were generated.")
    candidate_validation = build_control_week_validation(
        hourly_df,
        candidate_shift_plan,
        work_interval_lookup,
        group_to_members,
    )

    baseline_forecast = forecast_with_baseline_slots(forecast_df)
    baseline_hourly = None
    baseline_shift_plan = None
    if baseline_forecast is not None:
        baseline_hourly = expand_to_hourly_demand(
            baseline_forecast,
            payment_rates,
            delivery_p70_rates,
        )
        previous_minimize = MINIMIZE_SHIFT_OVERCOVERAGE
        MINIMIZE_SHIFT_OVERCOVERAGE = False
        try:
            baseline_shift_plan = build_business_shift_plan(
                baseline_hourly,
                work_interval_lookup,
                group_to_members,
            )
        finally:
            MINIMIZE_SHIFT_OVERCOVERAGE = previous_minimize

    actual_hours = load_actual_courier_hours(FORECAST_PATH)
    calibration_comparison, candidate_accepted = (
        build_courier_hour_calibration_comparison(
            baseline_shift_plan,
            candidate_shift_plan,
            actual_hours,
            candidate_validation,
        )
    )
    if candidate_accepted or baseline_shift_plan is None:
        shift_plan = candidate_shift_plan
    else:
        forecast_df = baseline_forecast
        hourly_df = baseline_hourly
        shift_plan = baseline_shift_plan
    shift_plan = attach_shift_delivery_p70(shift_plan, hourly_df)

    reports = build_reports(
        forecast_df,
        shift_plan,
        location_name_map,
        business_group_map,
        work_window_report,
    )
    reports["partner_absorb_audit"] = absorb_audit
    reports["partner_grouping_config_audit"] = build_grouping_config_audit(
        partner_grouping
    )
    reports["partner_absorb_summary"] = build_partner_absorb_summary(
        absorb_audit,
        location_name_map,
    )
    reports["shift_earnings_audit"] = build_shift_earnings_audit(shift_plan)
    reports["payment_input_quality_audit"] = payment_quality_audit
    reports["courier_hour_calibration_comparison"] = calibration_comparison
    reports["control_week_validation"] = build_control_week_validation(
        hourly_df,
        shift_plan,
        work_interval_lookup,
        group_to_members,
    )
    reports["hourly_courier_need_explanation"] = (
        build_hourly_courier_need_explanation(
            hourly_df,
            location_name_map,
            business_group_map,
            work_interval_lookup,
            group_to_members,
        )
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
