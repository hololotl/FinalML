import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from .partner_grouping import (
        apply_partner_grouping,
        build_grouping_config_audit,
        load_partner_grouping,
        normalize_location_id,
    )
except ImportError:
    from partner_grouping import (
        apply_partner_grouping,
        build_grouping_config_audit,
        load_partner_grouping,
        normalize_location_id,
    )


DB_URL = os.getenv(
    "DB_URL",
    "postgresql://courier:1337@localhost:1338/coffee",
)
TABLE_NAME = os.getenv("ORDERS_TABLE", "orders")
TIMESTAMP_COLUMN = os.getenv("ORDERS_TIMESTAMP_COLUMN", "delivering_at")
LOCATION_COLUMN = os.getenv("ORDERS_RESTAURANT_COLUMN", "location_id")
COURIER_COLUMN = os.getenv("ORDERS_COURIER_COLUMN", "courier_id")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PREDICTIONS_PATH = Path(
    os.getenv(
        "ORDERS_PREDICTIONS_PATH",
        PROJECT_DIR / "week_model" / "res" / "predictions_full.csv",
    )
)
OUTPUT_DIR = Path(os.getenv("COURIER_OUTPUT_DIR", SCRIPT_DIR / "res"))
DELIVERY_DURATION_PATH = Path(
    os.getenv(
        "DELIVERY_DURATION_PATH",
        OUTPUT_DIR
        / "delivery_duration"
        / "delivery_duration_by_planning_time_segment.csv",
    )
)

HISTORY_LOOKBACK_DAYS = int(os.getenv("COURIER_HISTORY_LOOKBACK_DAYS", "14"))
SAFETY_BUFFER = float(os.getenv("COURIER_SAFETY_BUFFER", "1.15"))
MIN_PREDICTED_ORDERS_FOR_SLOT = float(
    os.getenv(
        "MIN_PREDICTED_ORDERS_FOR_SLOT",
        os.getenv("MIN_PREDICTED_ORDERS_FOR_COURIER", "0.5"),
    )
)
DEFAULT_AUTO_SHARE = float(os.getenv("DEFAULT_AUTO_SHARE", "0.80"))
DEFAULT_AUTO_ORDERS_PER_SLOT_PER_HOUR = float(
    os.getenv("DEFAULT_AUTO_ORDERS_PER_SLOT_PER_HOUR", "1.5")
)
DEFAULT_BIKE_ORDERS_PER_SLOT_PER_HOUR = float(
    os.getenv("DEFAULT_BIKE_ORDERS_PER_SLOT_PER_HOUR", "1.0")
)
MIN_SLOT_EQUIV_FOR_PRODUCTIVITY = float(
    os.getenv(
        "MIN_SLOT_EQUIV_FOR_PRODUCTIVITY",
        os.getenv("MIN_COURIER_EQUIV_FOR_PRODUCTIVITY", "20"),
    )
)
MIN_ORDERS_FOR_PRODUCTIVITY = int(os.getenv("MIN_ORDERS_FOR_PRODUCTIVITY", "50"))
MIN_LOCATIONS_FOR_PRODUCTIVITY = int(os.getenv("MIN_LOCATIONS_FOR_PRODUCTIVITY", "5"))
PRODUCTIVITY_MIN = float(os.getenv("PRODUCTIVITY_MIN", "1.0"))
PRODUCTIVITY_MAX = float(os.getenv("PRODUCTIVITY_MAX", "30.0"))
SLOT_CAPACITY_MULTIPLIER = float(os.getenv("SLOT_CAPACITY_MULTIPLIER", "1.30"))
SLOT_ROUNDING_MODE = os.getenv("SLOT_ROUNDING_MODE", "combined")
JOINT_DAILY_ROUNDING = os.getenv("JOINT_DAILY_ROUNDING", "1") == "1"
CALIBRATED_SAFETY_BUFFER = float(
    os.getenv("CALIBRATED_SAFETY_BUFFER", "1.0")
)
SCHEDULE_SHARE_PRIOR_HOURS = float(
    os.getenv("SCHEDULE_SHARE_PRIOR_HOURS", "24")
)
CYCLE_TIME_PERCENTILE = os.getenv("CYCLE_TIME_PERCENTILE", "p80_minutes")
CYCLE_TIME_RETURN_MULTIPLIER = float(
    os.getenv("CYCLE_TIME_RETURN_MULTIPLIER", "2.0")
)
MIN_CYCLE_TIME_ORDERS = int(os.getenv("MIN_CYCLE_TIME_ORDERS", "30"))

AUTO = "auto"
BIKE = "bike"
VEHICLE_TYPES = [AUTO, BIKE]

TRANSPORT_AUTO_BIKE = 0
TRANSPORT_AUTO = 1
TRANSPORT_BIKE = 2

def build_engine():
    return create_engine(DB_URL)


def build_merged_location_groups():
    """Compatibility wrapper; grouping now comes only from partner_donor_map.csv."""
    return load_partner_grouping().location_to_planning


def build_group_to_members(location_to_group):
    grouping = load_partner_grouping()
    return grouping.planning_to_members


def load_location_transport(engine):
    transport_df = pd.read_sql_query(
        text(
            "SELECT id AS location_id, COALESCE(transport, 0) AS transport "
            "FROM locations"
        ),
        engine,
    )
    if transport_df.empty:
        return {}
    transport_df["location_id"] = transport_df["location_id"].astype(str)
    return {
        str(row.location_id): int(row.transport)
        for row in transport_df.itertuples(index=False)
    }


def resolve_location_transport(location_id, transport_by_location, group_to_members):
    location_id = str(location_id)
    if location_id.startswith("grp_"):
        member_transports = [
            int(transport_by_location.get(member_id, TRANSPORT_AUTO_BIKE))
            for member_id in group_to_members.get(location_id, [])
        ]
        if not member_transports:
            return TRANSPORT_AUTO_BIKE
        if all(value == TRANSPORT_AUTO for value in member_transports):
            return TRANSPORT_AUTO
        if all(value == TRANSPORT_BIKE for value in member_transports):
            return TRANSPORT_BIKE
        return TRANSPORT_AUTO_BIKE
    return int(transport_by_location.get(location_id, TRANSPORT_AUTO_BIKE))


def transport_allowed_vehicles(transport):
    if transport == TRANSPORT_AUTO:
        return {AUTO}
    if transport == TRANSPORT_BIKE:
        return {BIKE}
    return {AUTO, BIKE}


def compute_vehicle_slots(auto_orders, bike_orders, slot_capacity):
    if SLOT_ROUNDING_MODE == "combined":
        auto_raw_slots = raw_slot_demand(auto_orders, slot_capacity[AUTO])
        bike_raw_slots = raw_slot_demand(bike_orders, slot_capacity[BIKE])
        return allocate_slots_from_raw_demand(auto_raw_slots, bike_raw_slots)
    return (
        required_slots(auto_orders, slot_capacity[AUTO]),
        required_slots(bike_orders, slot_capacity[BIKE]),
    )


def apply_transport_to_slot_plan(
    transport,
    prediction,
    shares,
    slot_capacity,
):
    allowed = transport_allowed_vehicles(transport)
    prediction = max(float(prediction), 0.0)

    if allowed == {AUTO}:
        shares = {AUTO: 1.0, BIKE: 0.0}
        auto_orders = prediction
        bike_orders = 0.0
    elif allowed == {BIKE}:
        shares = {AUTO: 0.0, BIKE: 1.0}
        auto_orders = 0.0
        bike_orders = prediction
    else:
        auto_orders = prediction * shares[AUTO]
        bike_orders = prediction * shares[BIKE]

    auto_slots, bike_slots = compute_vehicle_slots(
        auto_orders,
        bike_orders,
        slot_capacity,
    )
    if AUTO not in allowed:
        auto_orders = 0.0
        auto_slots = 0
        shares[AUTO] = 0.0
    if BIKE not in allowed:
        bike_orders = 0.0
        bike_slots = 0
        shares[BIKE] = 0.0

    return shares, auto_orders, bike_orders, auto_slots, bike_slots


def enforce_transport_on_forecast(
    forecast_df,
    transport_by_location,
    group_to_members,
):
    """Zero disallowed vehicle columns without rebuilding slot demand."""
    if forecast_df.empty:
        return forecast_df

    forecast = forecast_df.copy()
    for idx, row in forecast.iterrows():
        transport = resolve_location_transport(
            row[LOCATION_COLUMN],
            transport_by_location,
            group_to_members,
        )
        allowed = transport_allowed_vehicles(transport)
        if allowed == {AUTO, BIKE}:
            continue

        auto_slots = int(row.get("auto_slots_needed", 0) or 0)
        bike_slots = int(row.get("bike_slots_needed", 0) or 0)
        auto_orders = float(row.get("auto_order_prediction", 0) or 0)
        bike_orders = float(row.get("bike_order_prediction", 0) or 0)

        if AUTO not in allowed:
            auto_slots = 0
            auto_orders = 0.0
        if BIKE not in allowed:
            bike_slots = 0
            bike_orders = 0.0

        total_orders = float(row.get("orders_prediction", 0) or 0)
        if allowed == {AUTO}:
            auto_orders = total_orders
            forecast.at[idx, "auto_order_share"] = 1.0
            forecast.at[idx, "bike_order_share"] = 0.0
        elif allowed == {BIKE}:
            bike_orders = total_orders
            forecast.at[idx, "auto_order_share"] = 0.0
            forecast.at[idx, "bike_order_share"] = 1.0

        forecast.at[idx, "auto_order_prediction"] = auto_orders
        forecast.at[idx, "bike_order_prediction"] = bike_orders
        forecast.at[idx, "auto_slots_needed"] = auto_slots
        forecast.at[idx, "bike_slots_needed"] = bike_slots
        forecast.at[idx, "total_slots_needed"] = auto_slots + bike_slots
        forecast.at[idx, "auto_couriers_needed"] = auto_slots
        forecast.at[idx, "bike_couriers_needed"] = bike_slots
        forecast.at[idx, "total_couriers_needed"] = auto_slots + bike_slots

    return forecast


def apply_location_grouping(df, location_to_group):
    df = df.copy()
    df[LOCATION_COLUMN] = (
        df[LOCATION_COLUMN]
        .map(normalize_location_id)
        .map(lambda x: location_to_group.get(x, x))
    )
    return df


def vehicle_to_type(vehicle):
    try:
        vehicle_int = int(vehicle)
    except (TypeError, ValueError):
        vehicle_int = 2
    if vehicle_int == 1:
        return BIKE
    return AUTO


def to_moscow_datetime(series):
    return (
        pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
        .dt.tz_convert("Europe/Moscow")
    )


def datetime_to_ms(dt_value):
    ts = pd.Timestamp(dt_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Moscow")
    return int(ts.tz_convert("UTC").timestamp() * 1000)


def classify_fine_segment(hour):
    if 0 <= hour < 6:
        return "deep_night"
    if 6 <= hour < 8:
        return "opening"
    if 8 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 11:
        return "morning"
    if 11 <= hour < 15:
        return "lunch"
    if 15 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "dinner"
    if 21 <= hour < 24:
        return "late_evening"
    return "other"


def classify_coarse_segment(hour):
    if 0 <= hour < 6:
        return "block_00_06"
    if 6 <= hour < 12:
        return "block_06_12"
    if 12 <= hour < 18:
        return "block_12_18"
    if 18 <= hour < 24:
        return "block_18_24"
    return "other"


def classify_time_segment(hour, restaurant_segment):
    if restaurant_segment in {"low", "medium"}:
        return classify_coarse_segment(hour)
    return classify_fine_segment(hour)


def segment_hours(time_segment):
    if str(time_segment).startswith("block_"):
        return 6.0
    return {
        "deep_night": 6.0,
        "opening": 2.0,
        "early_morning": 1.0,
        "morning": 2.0,
        "lunch": 4.0,
        "afternoon": 3.0,
        "dinner": 3.0,
        "late_evening": 3.0,
    }.get(time_segment, 1.0)


def default_orders_per_slot(time_segment, vehicle_type):
    if vehicle_type == AUTO:
        rate = DEFAULT_AUTO_ORDERS_PER_SLOT_PER_HOUR
    else:
        rate = DEFAULT_BIKE_ORDERS_PER_SLOT_PER_HOUR
    return float(
        np.clip(
            segment_hours(time_segment) * rate,
            PRODUCTIVITY_MIN,
            PRODUCTIVITY_MAX,
        )
    )


def segment_start_hours(restaurant_segment):
    if restaurant_segment in {"low", "medium"}:
        return [
            ("block_00_06", 0),
            ("block_06_12", 6),
            ("block_12_18", 12),
            ("block_18_24", 18),
        ]
    return [
        ("deep_night", 0),
        ("opening", 6),
        ("early_morning", 8),
        ("morning", 9),
        ("lunch", 11),
        ("afternoon", 15),
        ("dinner", 18),
        ("late_evening", 21),
    ]


def load_predictions(path):
    pred_df = pd.read_csv(path)
    pred_df[LOCATION_COLUMN] = pred_df[LOCATION_COLUMN].astype(str)
    pred_df["segment_datetime"] = pd.to_datetime(
        pred_df["segment_datetime"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert("Europe/Moscow")
    pred_df = pred_df.dropna(subset=["segment_datetime", "prediction"])
    pred_df["segment_hours"] = pred_df["time_segment"].map(segment_hours)
    pred_df["segment_end"] = pred_df["segment_datetime"] + pd.to_timedelta(
        pred_df["segment_hours"],
        unit="h",
    )
    return pred_df


def aggregate_predictions_to_planning_locations(pred_df, grouping):
    if pred_df.empty:
        return pred_df.copy(), pd.DataFrame()

    source = pred_df.copy()
    source[LOCATION_COLUMN] = source[LOCATION_COLUMN].map(normalize_location_id)
    legacy_groups = source[LOCATION_COLUMN].str.startswith("grp_")
    if legacy_groups.any():
        legacy_ids = sorted(source.loc[legacy_groups, LOCATION_COLUMN].unique())
        raise ValueError(
            "Forecast contains legacy grp_* location IDs. Retrain and regenerate "
            f"week_model with partner_donor_map.csv first: {legacy_ids[:10]}"
        )

    source["source_location_id"] = source[LOCATION_COLUMN]
    source["planning_location_id"] = source["source_location_id"].map(
        grouping.planning_location_id
    )
    source["is_absorbed_partner"] = (
        source["source_location_id"] != source["planning_location_id"]
    )

    partner_names = grouping.partner_name_by_id
    source["absorbed_partner_name"] = source["source_location_id"].map(
        lambda location_id: partner_names.get(location_id, location_id)
    )

    audit = source.loc[source["is_absorbed_partner"]].copy()
    if audit.empty:
        audit_df = pd.DataFrame(
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
    else:
        audit_df = audit[
            [
                "source_location_id",
                "absorbed_partner_name",
                "planning_location_id",
                "segment_datetime",
                "time_segment",
                "prediction",
            ]
        ].rename(
            columns={
                "absorbed_partner_name": "partner_name",
                "planning_location_id": "kfm_donor_id",
                "prediction": "orders_moved",
            }
        )
        audit_df["action"] = "group_before_slot_calculation"

    segment_rank = {"low": 0, "medium": 1, "high": 2, "mega": 3}
    source["_segment_rank"] = source["segment"].map(segment_rank).fillna(-1)
    source["_donor_segment_rank"] = np.where(
        source["source_location_id"] == source["planning_location_id"],
        source["_segment_rank"],
        -1,
    )
    source["_absorbed_orders"] = np.where(
        source["is_absorbed_partner"],
        source["prediction"].clip(lower=0),
        0.0,
    )
    source["_absorbed_name"] = np.where(
        source["is_absorbed_partner"],
        source["absorbed_partner_name"],
        "",
    )

    group_cols = ["planning_location_id", "segment_datetime", "time_segment"]
    aggregation = {
        "prediction": "sum",
        "_segment_rank": "max",
        "_donor_segment_rank": "max",
        "_absorbed_orders": "sum",
        "_absorbed_name": lambda values: ",".join(
            dict.fromkeys(str(value) for value in values if str(value))
        ),
    }
    if "orders_count" in source.columns:
        aggregation["orders_count"] = lambda values: values.sum(min_count=1)

    grouped = source.groupby(group_cols, as_index=False).agg(aggregation)
    grouped = grouped.rename(
        columns={
            "planning_location_id": LOCATION_COLUMN,
            "_absorbed_orders": "absorbed_partner_orders",
            "_absorbed_name": "absorbed_partners",
        }
    )
    rank_to_segment = {value: key for key, value in segment_rank.items()}
    grouped["_resolved_segment_rank"] = np.where(
        grouped["_donor_segment_rank"] >= 0,
        grouped["_donor_segment_rank"],
        grouped["_segment_rank"],
    )
    grouped["segment"] = grouped.pop("_resolved_segment_rank").map(rank_to_segment)
    grouped = grouped.drop(columns=["_segment_rank", "_donor_segment_rank"])
    grouped["segment_hours"] = grouped["time_segment"].map(segment_hours)
    grouped["segment_end"] = grouped["segment_datetime"] + pd.to_timedelta(
        grouped["segment_hours"],
        unit="h",
    )
    configured_partner_names = {
        planning_id: ",".join(
            grouping.partner_name_by_id.get(member_id, member_id)
            for member_id in members
            if member_id != planning_id
        )
        for planning_id, members in grouping.planning_to_members.items()
    }
    grouped["absorbed_partners"] = grouped.apply(
        lambda row: row["absorbed_partners"]
        or configured_partner_names.get(str(row[LOCATION_COLUMN]), ""),
        axis=1,
    )
    return grouped, audit_df


def build_location_segment_map(pred_df):
    return (
        pred_df[[LOCATION_COLUMN, "segment"]]
        .dropna()
        .drop_duplicates(subset=[LOCATION_COLUMN])
        .set_index(LOCATION_COLUMN)["segment"]
        .to_dict()
    )


def build_historical_segment_grid(location_segment_map, history_start, history_finish):
    rows = []
    start_day = pd.Timestamp(history_start).floor("D")
    finish_day = pd.Timestamp(history_finish).floor("D")
    for location_id, restaurant_segment in location_segment_map.items():
        for date in pd.date_range(start_day, finish_day, freq="D"):
            for time_segment, start_hour in segment_start_hours(restaurant_segment):
                segment_datetime = date + pd.Timedelta(hours=start_hour)
                hours = segment_hours(time_segment)
                segment_end = segment_datetime + pd.Timedelta(hours=hours)
                if segment_end <= history_start or segment_datetime >= history_finish:
                    continue
                rows.append({
                    LOCATION_COLUMN: str(location_id),
                    "segment": restaurant_segment,
                    "time_segment": time_segment,
                    "segment_datetime": segment_datetime,
                    "segment_end": segment_end,
                    "segment_hours": hours,
                })
    return pd.DataFrame(rows)


def load_order_vehicle_history(engine, history_start_ms, history_finish_ms):
    sql = f"""
        SELECT
            o.{LOCATION_COLUMN} AS {LOCATION_COLUMN},
            o.{TIMESTAMP_COLUMN} AS {TIMESTAMP_COLUMN},
            COALESCE(c.vehicle, 2) AS vehicle
        FROM {TABLE_NAME} o
        LEFT JOIN couriers c
            ON o.{COURIER_COLUMN} = c.id
        WHERE o.{TIMESTAMP_COLUMN} >= :history_start_ms
          AND o.{TIMESTAMP_COLUMN} < :history_finish_ms
    """
    return pd.read_sql_query(
        text(sql),
        engine,
        params={
            "history_start_ms": history_start_ms,
            "history_finish_ms": history_finish_ms,
        },
    )


def load_schedule_history(engine, history_start_ms, history_finish_ms):
    sql = """
        SELECT
            cs.id AS schedule_id,
            cs.start AS schedule_start,
            cs.finish AS schedule_finish,
            csl.location_id AS location_id,
            COALESCE(c.vehicle, 2) AS vehicle
        FROM courier_schedule cs
        JOIN courier_shifts sh
            ON cs.shift_id = sh.id
        JOIN courier_schedule_locations csl
            ON csl.schedule_id = cs.id
        LEFT JOIN couriers c
            ON sh.courier_id = c.id
        WHERE cs.finish > :history_start_ms
          AND cs.start < :history_finish_ms
          AND COALESCE(sh.no_show_confirmed, false) = false
    """
    return pd.read_sql_query(
        text(sql),
        engine,
        params={
            "history_start_ms": history_start_ms,
            "history_finish_ms": history_finish_ms,
        },
    )


def prepare_order_history(order_df, location_to_group, location_segment_map):
    if order_df.empty:
        return order_df
    order_df = apply_location_grouping(order_df, location_to_group)
    order_df["datetime"] = to_moscow_datetime(order_df[TIMESTAMP_COLUMN])
    order_df = order_df.dropna(subset=["datetime"])
    order_df["hour"] = order_df["datetime"].dt.hour
    order_df["date"] = order_df["datetime"].dt.floor("D")
    order_df["vehicle_type"] = order_df["vehicle"].map(vehicle_to_type)
    order_df["segment"] = (
        order_df[LOCATION_COLUMN]
        .astype(str)
        .map(location_segment_map)
        .fillna("high")
    )
    order_df["time_segment"] = order_df.apply(
        lambda row: classify_time_segment(row["hour"], row["segment"]),
        axis=1,
    )
    return order_df


def build_order_share_tables(order_df):
    if order_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, {AUTO: DEFAULT_AUTO_SHARE, BIKE: 1 - DEFAULT_AUTO_SHARE}

    order_counts = (
        order_df.groupby([LOCATION_COLUMN, "segment", "time_segment", "vehicle_type"])
        .size()
        .reset_index(name="orders")
    )
    order_totals = (
        order_counts.groupby([LOCATION_COLUMN, "time_segment"])["orders"]
        .sum()
        .reset_index(name="total_orders")
    )
    location_share = order_counts.merge(
        order_totals,
        on=[LOCATION_COLUMN, "time_segment"],
        how="left",
    )
    location_share["vehicle_share"] = (
        location_share["orders"] / location_share["total_orders"].replace(0, np.nan)
    )

    segment_counts = (
        order_df.groupby(["segment", "time_segment", "vehicle_type"])
        .size()
        .reset_index(name="orders")
    )
    segment_totals = (
        segment_counts.groupby(["segment", "time_segment"])["orders"]
        .sum()
        .reset_index(name="total_orders")
    )
    segment_share = segment_counts.merge(
        segment_totals,
        on=["segment", "time_segment"],
        how="left",
    )
    segment_share["vehicle_share"] = (
        segment_share["orders"] / segment_share["total_orders"].replace(0, np.nan)
    )

    global_counts = order_df.groupby("vehicle_type").size()
    total_orders = global_counts.sum()
    global_share = {
        vehicle_type: float(global_counts.get(vehicle_type, 0) / total_orders)
        for vehicle_type in VEHICLE_TYPES
    }
    if sum(global_share.values()) == 0:
        global_share = {AUTO: DEFAULT_AUTO_SHARE, BIKE: 1 - DEFAULT_AUTO_SHARE}

    return order_counts, location_share, segment_share, global_share


def prepare_schedule_history(schedule_df, location_to_group):
    if schedule_df.empty:
        return schedule_df
    schedule_df = schedule_df.copy()
    schedule_df[LOCATION_COLUMN] = (
        schedule_df[LOCATION_COLUMN]
        .map(normalize_location_id)
        .map(lambda x: location_to_group.get(x, x))
    )
    schedule_df["vehicle_type"] = schedule_df["vehicle"].map(vehicle_to_type)
    schedule_df["schedule_start_dt"] = to_moscow_datetime(schedule_df["schedule_start"])
    schedule_df["schedule_finish_dt"] = to_moscow_datetime(schedule_df["schedule_finish"])
    schedule_df = schedule_df.dropna(subset=["schedule_start_dt", "schedule_finish_dt"])
    schedule_df = schedule_df.drop_duplicates(
        subset=["schedule_id", LOCATION_COLUMN, "vehicle_type"]
    )
    location_count = (
        schedule_df.groupby("schedule_id")[LOCATION_COLUMN]
        .nunique()
        .reset_index(name="schedule_location_count")
    )
    schedule_df = schedule_df.merge(location_count, on="schedule_id", how="left")
    schedule_df["location_allocation"] = (
        1.0 / schedule_df["schedule_location_count"].clip(lower=1)
    )
    return schedule_df


def build_courier_equivalent_by_segment(schedule_df, pred_df):
    if schedule_df.empty:
        return pd.DataFrame(
            columns=[
                LOCATION_COLUMN,
                "segment",
                "time_segment",
                "vehicle_type",
                "courier_equiv",
                "shift_overlap_hours",
            ]
        )

    grid_cols = [
        LOCATION_COLUMN,
        "segment",
        "time_segment",
        "segment_datetime",
        "segment_end",
        "segment_hours",
    ]
    segment_grid_by_location = {
        location_id: group[grid_cols].drop_duplicates().reset_index(drop=True)
        for location_id, group in pred_df.groupby(LOCATION_COLUMN)
    }

    rows = []
    for row in schedule_df.itertuples(index=False):
        location_id = getattr(row, LOCATION_COLUMN)
        grid = segment_grid_by_location.get(str(location_id))
        if grid is None or grid.empty:
            continue
        overlap_mask = (
            (grid["segment_datetime"] < row.schedule_finish_dt)
            & (grid["segment_end"] > row.schedule_start_dt)
        )
        if not overlap_mask.any():
            continue
        overlap_df = grid.loc[overlap_mask].copy()
        overlap_start = overlap_df["segment_datetime"].where(
            overlap_df["segment_datetime"] > row.schedule_start_dt,
            row.schedule_start_dt,
        )
        overlap_finish = overlap_df["segment_end"].where(
            overlap_df["segment_end"] < row.schedule_finish_dt,
            row.schedule_finish_dt,
        )
        overlap_hours = (
            (overlap_finish - overlap_start).dt.total_seconds().clip(lower=0)
            / 3600.0
        )
        courier_equiv = (
            overlap_hours
            / overlap_df["segment_hours"].replace(0, np.nan)
            * row.location_allocation
        )
        for idx, overlap_row in overlap_df.iterrows():
            rows.append({
                LOCATION_COLUMN: location_id,
                "segment": overlap_row["segment"],
                "time_segment": overlap_row["time_segment"],
                "vehicle_type": row.vehicle_type,
                "shift_overlap_hours": float(overlap_hours.loc[idx]),
                "courier_equiv": float(courier_equiv.loc[idx]),
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                LOCATION_COLUMN,
                "segment",
                "time_segment",
                "vehicle_type",
                "courier_equiv",
                "shift_overlap_hours",
            ]
        )

    return (
        pd.DataFrame(rows)
        .groupby([LOCATION_COLUMN, "segment", "time_segment", "vehicle_type"], as_index=False)
        .agg(
            courier_equiv=("courier_equiv", "sum"),
            shift_overlap_hours=("shift_overlap_hours", "sum"),
        )
    )


def build_slot_equivalent_by_window(schedule_df, segment_grid):
    empty_columns = [
        LOCATION_COLUMN,
        "segment_datetime",
        "segment",
        "time_segment",
        "vehicle_type",
        "actual_slots",
        "shift_overlap_hours",
    ]
    if schedule_df.empty or segment_grid.empty:
        return pd.DataFrame(columns=empty_columns)

    grid_cols = [
        LOCATION_COLUMN,
        "segment",
        "time_segment",
        "segment_datetime",
        "segment_end",
        "segment_hours",
    ]
    segment_grid_by_location = {
        location_id: group[grid_cols].drop_duplicates().reset_index(drop=True)
        for location_id, group in segment_grid.groupby(LOCATION_COLUMN)
    }

    rows = []
    for row in schedule_df.itertuples(index=False):
        location_id = getattr(row, LOCATION_COLUMN)
        grid = segment_grid_by_location.get(str(location_id))
        if grid is None or grid.empty:
            continue
        overlap_mask = (
            (grid["segment_datetime"] < row.schedule_finish_dt)
            & (grid["segment_end"] > row.schedule_start_dt)
        )
        if not overlap_mask.any():
            continue
        overlap_df = grid.loc[overlap_mask].copy()
        overlap_start = overlap_df["segment_datetime"].where(
            overlap_df["segment_datetime"] > row.schedule_start_dt,
            row.schedule_start_dt,
        )
        overlap_finish = overlap_df["segment_end"].where(
            overlap_df["segment_end"] < row.schedule_finish_dt,
            row.schedule_finish_dt,
        )
        overlap_hours = (
            (overlap_finish - overlap_start).dt.total_seconds().clip(lower=0)
            / 3600.0
        )
        for idx, overlap_row in overlap_df.iterrows():
            rows.append({
                LOCATION_COLUMN: location_id,
                "segment_datetime": overlap_row["segment_datetime"],
                "segment": overlap_row["segment"],
                "time_segment": overlap_row["time_segment"],
                "vehicle_type": row.vehicle_type,
                "shift_overlap_hours": float(overlap_hours.loc[idx]),
                "actual_slots": 1.0,
            })

    if not rows:
        return pd.DataFrame(columns=empty_columns)

    return (
        pd.DataFrame(rows)
        .groupby(
            [
                LOCATION_COLUMN,
                "segment_datetime",
                "segment",
                "time_segment",
                "vehicle_type",
            ],
            as_index=False,
        )
        .agg(
            actual_slots=("actual_slots", "sum"),
            shift_overlap_hours=("shift_overlap_hours", "sum"),
        )
    )


def build_productivity_tables(order_counts, courier_equiv_df):
    if order_counts.empty or courier_equiv_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, {}

    productivity = order_counts.merge(
        courier_equiv_df,
        on=[LOCATION_COLUMN, "segment", "time_segment", "vehicle_type"],
        how="outer",
    )
    productivity["orders"] = productivity["orders"].fillna(0)
    productivity["courier_equiv"] = productivity["courier_equiv"].fillna(0)
    productivity["orders_per_slot"] = (
        productivity["orders"]
        / productivity["courier_equiv"].replace(0, np.nan)
    )
    usable = productivity[
        (productivity["courier_equiv"] >= MIN_SLOT_EQUIV_FOR_PRODUCTIVITY)
        & (productivity["orders"] >= MIN_ORDERS_FOR_PRODUCTIVITY)
    ].copy()
    usable["orders_per_slot"] = usable["orders_per_slot"].clip(
        PRODUCTIVITY_MIN,
        PRODUCTIVITY_MAX,
    )

    segment_productivity = (
        productivity.groupby(["segment", "time_segment", "vehicle_type"], as_index=False)
        .agg(
            orders=("orders", "sum"),
            courier_equiv=("courier_equiv", "sum"),
            location_count=(LOCATION_COLUMN, "nunique"),
        )
    )
    segment_productivity["orders_per_slot"] = (
        segment_productivity["orders"]
        / segment_productivity["courier_equiv"].replace(0, np.nan)
    )
    segment_productivity = segment_productivity[
        (segment_productivity["courier_equiv"] >= MIN_SLOT_EQUIV_FOR_PRODUCTIVITY)
        & (segment_productivity["orders"] >= MIN_ORDERS_FOR_PRODUCTIVITY)
        & (segment_productivity["location_count"] >= MIN_LOCATIONS_FOR_PRODUCTIVITY)
    ].copy()
    segment_productivity["orders_per_slot"] = segment_productivity[
        "orders_per_slot"
    ].clip(PRODUCTIVITY_MIN, PRODUCTIVITY_MAX)

    time_productivity = (
        productivity.groupby(["time_segment", "vehicle_type"], as_index=False)
        .agg(
            orders=("orders", "sum"),
            courier_equiv=("courier_equiv", "sum"),
            location_count=(LOCATION_COLUMN, "nunique"),
        )
    )
    time_productivity["orders_per_slot"] = (
        time_productivity["orders"]
        / time_productivity["courier_equiv"].replace(0, np.nan)
    )
    time_productivity = time_productivity[
        (time_productivity["courier_equiv"] >= MIN_SLOT_EQUIV_FOR_PRODUCTIVITY)
        & (time_productivity["orders"] >= MIN_ORDERS_FOR_PRODUCTIVITY)
        & (time_productivity["location_count"] >= MIN_LOCATIONS_FOR_PRODUCTIVITY)
    ].copy()
    time_productivity["orders_per_slot"] = time_productivity[
        "orders_per_slot"
    ].clip(PRODUCTIVITY_MIN, PRODUCTIVITY_MAX)

    global_productivity = {}

    return usable, segment_productivity, time_productivity, global_productivity


def _weighted_duration_table(df, group_columns, percentile_column):
    if df.empty:
        return pd.DataFrame(
            columns=group_columns
            + ["duration_minutes", "duration_orders"]
        )
    rows = df.copy()
    rows["_weighted_duration"] = (
        rows[percentile_column] * rows["orders_count"]
    )
    result = (
        rows.groupby(group_columns, as_index=False)
        .agg(
            weighted_duration=("_weighted_duration", "sum"),
            duration_orders=("orders_count", "sum"),
        )
    )
    result["duration_minutes"] = (
        result.pop("weighted_duration")
        / result["duration_orders"].replace(0, np.nan)
    )
    return result


def build_cycle_time_tables(duration_df, percentile_column=CYCLE_TIME_PERCENTILE):
    empty = pd.DataFrame()
    if duration_df is None or duration_df.empty:
        return empty, empty, empty, empty
    required = {
        "planning_location_id",
        "segment",
        "time_segment",
        "vehicle_type",
        "orders_count",
        percentile_column,
    }
    missing = sorted(required - set(duration_df.columns))
    if missing:
        raise ValueError(f"Delivery duration report is missing columns: {missing}")

    duration = duration_df.copy()
    duration["planning_location_id"] = duration["planning_location_id"].map(
        normalize_location_id
    )
    duration["orders_count"] = pd.to_numeric(
        duration["orders_count"], errors="coerce"
    ).fillna(0)
    duration[percentile_column] = pd.to_numeric(
        duration[percentile_column], errors="coerce"
    )
    duration = duration[
        (duration["orders_count"] > 0)
        & (duration[percentile_column] > 0)
    ].copy()

    location = _weighted_duration_table(
        duration,
        [
            "planning_location_id",
            "segment",
            "time_segment",
            "vehicle_type",
        ],
        percentile_column,
    )
    location = location[
        location["duration_orders"] >= MIN_CYCLE_TIME_ORDERS
    ].copy()
    segment = _weighted_duration_table(
        duration,
        ["segment", "time_segment", "vehicle_type"],
        percentile_column,
    )
    time_segment = _weighted_duration_table(
        duration,
        ["time_segment", "vehicle_type"],
        percentile_column,
    )
    global_vehicle = _weighted_duration_table(
        duration,
        ["vehicle_type"],
        percentile_column,
    )
    return location, segment, time_segment, global_vehicle


def load_cycle_time_tables(path=DELIVERY_DURATION_PATH):
    path = Path(path)
    if not path.exists():
        return (pd.DataFrame(),) * 4
    return build_cycle_time_tables(pd.read_csv(path, dtype=str))


def lookup_cycle_time(
    row,
    vehicle_type,
    location_lookup,
    segment_lookup,
    time_lookup,
    global_lookup,
):
    keys = [
        (
            location_lookup,
            (
                str(row[LOCATION_COLUMN]),
                row["segment"],
                row["time_segment"],
                vehicle_type,
            ),
            "location_cycle_time",
        ),
        (
            segment_lookup,
            (row["segment"], row["time_segment"], vehicle_type),
            "segment_cycle_time",
        ),
        (
            time_lookup,
            (row["time_segment"], vehicle_type),
            "time_segment_cycle_time",
        ),
        (
            global_lookup,
            (vehicle_type,),
            "global_cycle_time",
        ),
    ]
    for lookup, key, source in keys:
        if key in lookup:
            duration_minutes, orders_count = lookup[key]
            return float(duration_minutes), int(orders_count), source
    return np.nan, 0, "missing_cycle_time"


def make_lookup(df, key_cols, value_col):
    if df.empty:
        return {}
    return {
        tuple(row[col] for col in key_cols): float(row[value_col])
        for _, row in df.iterrows()
    }


def make_duration_lookup(df, key_cols):
    if df.empty:
        return {}
    return {
        tuple(str(row[col]) if col == "planning_location_id" else row[col]
              for col in key_cols): (
            float(row["duration_minutes"]),
            int(row["duration_orders"]),
        )
        for _, row in df.iterrows()
    }


def build_location_overall_share_lookup(order_df):
    if order_df.empty:
        return {}

    grouped = (
        order_df.assign(**{LOCATION_COLUMN: order_df[LOCATION_COLUMN].astype(str)})
        .groupby([LOCATION_COLUMN, "vehicle_type"], as_index=False)
        .size()
        .rename(columns={"size": "orders"})
    )
    lookup = {}
    for location_id, group in grouped.groupby(LOCATION_COLUMN):
        total = float(group["orders"].sum())
        if total <= 0:
            continue
        for row in group.itertuples(index=False):
            lookup[(str(location_id), row.vehicle_type)] = float(row.orders / total)
    return lookup


def build_schedule_vehicle_share_lookup(schedule_df):
    """Estimate the operational vehicle mix from scheduled courier-hours.

    Mixed locations are shrunk towards the network mix so a small sample does
    not swing the forecast. Locations which historically use only one allowed
    vehicle remain locked to that vehicle.
    """
    if schedule_df.empty:
        return {}

    schedule = schedule_df.copy()
    duration_hours = (
        schedule["schedule_finish_dt"] - schedule["schedule_start_dt"]
    ).dt.total_seconds().clip(lower=0) / 3600.0
    allocation = schedule.get(
        "location_allocation",
        pd.Series(1.0, index=schedule.index),
    )
    schedule["courier_hours"] = duration_hours * allocation
    counts = (
        schedule.groupby(
            [schedule[LOCATION_COLUMN].astype(str), "vehicle_type"],
            as_index=False,
        )["courier_hours"]
        .sum()
    )
    global_hours = counts.groupby("vehicle_type")["courier_hours"].sum()
    global_total = float(global_hours.sum())
    global_share = {
        vehicle_type: (
            float(global_hours.get(vehicle_type, 0.0) / global_total)
            if global_total > 0
            else (DEFAULT_AUTO_SHARE if vehicle_type == AUTO else 1 - DEFAULT_AUTO_SHARE)
        )
        for vehicle_type in VEHICLE_TYPES
    }
    lookup = {}
    for location_id, group in counts.groupby(LOCATION_COLUMN):
        auto_hours = float(
            group.loc[group["vehicle_type"] == AUTO, "courier_hours"].sum()
        )
        bike_hours = float(
            group.loc[group["vehicle_type"] == BIKE, "courier_hours"].sum()
        )
        total_hours = auto_hours + bike_hours
        if total_hours <= 0:
            continue
        if bike_hours == 0 and auto_hours > 0:
            lookup[str(location_id)] = {AUTO: 1.0, BIKE: 0.0}
        elif auto_hours == 0 and bike_hours > 0:
            lookup[str(location_id)] = {AUTO: 0.0, BIKE: 1.0}
        else:
            denominator = total_hours + SCHEDULE_SHARE_PRIOR_HOURS
            lookup[str(location_id)] = {
                AUTO: (
                    auto_hours
                    + SCHEDULE_SHARE_PRIOR_HOURS * global_share[AUTO]
                )
                / denominator,
                BIKE: (
                    bike_hours
                    + SCHEDULE_SHARE_PRIOR_HOURS * global_share[BIKE]
                )
                / denominator,
            }
    return lookup


def lookup_vehicle_share(
    row,
    vehicle_type,
    location_share_lookup,
    segment_share_lookup,
    global_share,
    location_overall_share_lookup=None,
):
    location_id = str(row[LOCATION_COLUMN])
    key = (location_id, row["time_segment"], vehicle_type)
    if key in location_share_lookup:
        return location_share_lookup[key]
    if location_overall_share_lookup:
        overall_key = (location_id, vehicle_type)
        if overall_key in location_overall_share_lookup:
            return location_overall_share_lookup[overall_key]
    segment_key = (row["segment"], row["time_segment"], vehicle_type)
    if segment_key in segment_share_lookup:
        return segment_share_lookup[segment_key]
    return global_share.get(vehicle_type, 0.0)


def lookup_productivity(
    row,
    vehicle_type,
    location_productivity_lookup,
    segment_productivity_lookup,
    time_productivity_lookup,
    global_productivity,
):
    key = (row[LOCATION_COLUMN], row["segment"], row["time_segment"], vehicle_type)
    if key in location_productivity_lookup:
        return location_productivity_lookup[key], "location_shift_history"
    segment_key = (row["segment"], row["time_segment"], vehicle_type)
    if segment_key in segment_productivity_lookup:
        return segment_productivity_lookup[segment_key], "segment_shift_history"
    time_key = (row["time_segment"], vehicle_type)
    if time_key in time_productivity_lookup:
        return time_productivity_lookup[time_key], "time_segment_shift_history"
    if vehicle_type in global_productivity:
        return global_productivity[vehicle_type], "global_shift_history"
    return default_orders_per_slot(row["time_segment"], vehicle_type), "default_slot_capacity"


def required_slots(predicted_orders, orders_per_slot):
    if predicted_orders < MIN_PREDICTED_ORDERS_FOR_SLOT:
        return 0
    return int(np.ceil((predicted_orders / max(orders_per_slot, 1e-6)) * SAFETY_BUFFER))


def raw_slot_demand(predicted_orders, orders_per_slot):
    if predicted_orders < MIN_PREDICTED_ORDERS_FOR_SLOT:
        return 0.0
    return float(predicted_orders / max(orders_per_slot, 1e-6))


def cycle_time_slot_demand(predicted_orders, duration_minutes, window_hours):
    if (
        predicted_orders < MIN_PREDICTED_ORDERS_FOR_SLOT
        or not np.isfinite(duration_minutes)
        or duration_minutes <= 0
        or window_hours <= 0
    ):
        return 0.0
    cycle_minutes = duration_minutes * CYCLE_TIME_RETURN_MULTIPLIER
    return float(predicted_orders * cycle_minutes / (60.0 * window_hours))


def select_slot_demand(
    auto_orders,
    bike_orders,
    slot_capacity,
    cycle_minutes,
    window_hours,
):
    order_by_vehicle = {AUTO: auto_orders, BIKE: bike_orders}
    history_raw = {
        vehicle_type: raw_slot_demand(
            order_by_vehicle[vehicle_type],
            slot_capacity[vehicle_type],
        )
        for vehicle_type in VEHICLE_TYPES
    }
    cycle_raw = {
        vehicle_type: cycle_time_slot_demand(
            order_by_vehicle[vehicle_type],
            cycle_minutes.get(vehicle_type, np.nan),
            window_hours,
        )
        for vehicle_type in VEHICLE_TYPES
    }
    selected_raw = {
        vehicle_type: max(
            history_raw[vehicle_type],
            cycle_raw[vehicle_type],
        )
        for vehicle_type in VEHICLE_TYPES
    }
    if SLOT_ROUNDING_MODE == "combined":
        auto_slots, bike_slots = allocate_slots_from_raw_demand(
            selected_raw[AUTO],
            selected_raw[BIKE],
        )
    else:
        auto_slots = int(np.ceil(selected_raw[AUTO] * SAFETY_BUFFER))
        bike_slots = int(np.ceil(selected_raw[BIKE] * SAFETY_BUFFER))
    selected_source = {
        vehicle_type: (
            "cycle_time"
            if cycle_raw[vehicle_type] > history_raw[vehicle_type]
            else "history"
        )
        for vehicle_type in VEHICLE_TYPES
    }
    return history_raw, cycle_raw, selected_raw, selected_source, auto_slots, bike_slots


def allocate_slots_from_raw_demand(raw_auto_slots, raw_bike_slots):
    total_raw_slots = raw_auto_slots + raw_bike_slots
    if total_raw_slots <= 0:
        return 0, 0

    total_slots = int(np.ceil(total_raw_slots * SAFETY_BUFFER))
    if SLOT_ROUNDING_MODE != "combined":
        return (
            int(np.ceil(raw_auto_slots * SAFETY_BUFFER)),
            int(np.ceil(raw_bike_slots * SAFETY_BUFFER)),
        )

    auto_float = total_slots * raw_auto_slots / total_raw_slots
    auto_slots = int(np.floor(auto_float))
    bike_slots = total_slots - auto_slots

    if raw_auto_slots <= 0:
        return 0, total_slots
    if raw_bike_slots <= 0:
        return total_slots, 0

    return auto_slots, bike_slots


def apply_joint_daily_rounding(forecast_df):
    """Round daily workload jointly instead of ceiling every weak block.

    The daily sum is conserved after the safety buffer. Integer slots are
    assigned to blocks with the largest fractional workload first, then split
    between vehicles according to their continuous demand.
    """
    if forecast_df.empty or not JOINT_DAILY_ROUNDING:
        return forecast_df

    result = forecast_df.copy()
    result["_rounding_date"] = pd.to_datetime(
        result["segment_datetime"]
    ).dt.date.astype(str)
    result["_total_scaled_raw"] = (
        result["auto_selected_raw_slots"] + result["bike_selected_raw_slots"]
    ) * CALIBRATED_SAFETY_BUFFER
    result["calibrated_total_slots"] = 0

    group_columns = [LOCATION_COLUMN, "_rounding_date"]
    for _, index in result.groupby(group_columns, sort=False).groups.items():
        group_index = list(index)
        values = result.loc[group_index, "_total_scaled_raw"].clip(lower=0)
        floors = np.floor(values).astype(int)
        daily_target = int(np.ceil(values.sum()))
        extras = max(daily_target - int(floors.sum()), 0)
        fractions = values - floors
        ranked = fractions.sort_values(ascending=False, kind="stable").index
        totals = floors.copy()
        if extras:
            totals.loc[ranked[:extras]] += 1
        result.loc[group_index, "calibrated_total_slots"] = totals

    result["auto_slots_needed"] = 0
    result["bike_slots_needed"] = 0
    for _, index in result.groupby(group_columns, sort=False).groups.items():
        group_index = list(index)
        totals = result.loc[group_index, "calibrated_total_slots"].astype(int)
        raw_total = (
            result.loc[group_index, "auto_selected_raw_slots"]
            + result.loc[group_index, "bike_selected_raw_slots"]
        )
        desired_auto = totals * (
            result.loc[group_index, "auto_selected_raw_slots"]
            / raw_total.replace(0, np.nan)
        ).fillna(0.0)
        auto_slots = np.floor(desired_auto).astype(int)
        target_auto = int(np.rint(desired_auto.sum()))
        auto_extras = max(target_auto - int(auto_slots.sum()), 0)
        auto_ranked = (desired_auto - auto_slots).sort_values(
            ascending=False,
            kind="stable",
        ).index
        if auto_extras:
            auto_slots.loc[auto_ranked[:auto_extras]] += 1
        auto_slots = np.minimum(auto_slots, totals)
        result.loc[group_index, "auto_slots_needed"] = auto_slots
        result.loc[group_index, "bike_slots_needed"] = totals - auto_slots

    result["auto_slots_needed"] = result["auto_slots_needed"].astype(int)
    result["bike_slots_needed"] = result["bike_slots_needed"].astype(int)
    result["total_slots_needed"] = (
        result["auto_slots_needed"] + result["bike_slots_needed"]
    )
    result["auto_couriers_needed"] = result["auto_slots_needed"]
    result["bike_couriers_needed"] = result["bike_slots_needed"]
    result["total_couriers_needed"] = result["total_slots_needed"]
    result["slot_rounding_strategy"] = "joint_location_day"
    result["baseline_safety_buffer"] = SAFETY_BUFFER
    result["safety_buffer"] = CALIBRATED_SAFETY_BUFFER
    return result.drop(columns=["_rounding_date", "_total_scaled_raw"])


def build_courier_forecast(
    pred_df,
    location_share,
    segment_share,
    global_share,
    location_productivity,
    segment_productivity,
    time_productivity,
    global_productivity,
    transport_by_location=None,
    group_to_members=None,
    location_overall_share_lookup=None,
    schedule_vehicle_share_lookup=None,
    cycle_location=None,
    cycle_segment=None,
    cycle_time_segment=None,
    cycle_global=None,
):
    location_share_lookup = make_lookup(
        location_share,
        [LOCATION_COLUMN, "time_segment", "vehicle_type"],
        "vehicle_share",
    )
    segment_share_lookup = make_lookup(
        segment_share,
        ["segment", "time_segment", "vehicle_type"],
        "vehicle_share",
    )
    location_productivity_lookup = make_lookup(
        location_productivity,
        [LOCATION_COLUMN, "segment", "time_segment", "vehicle_type"],
        "orders_per_slot",
    )
    segment_productivity_lookup = make_lookup(
        segment_productivity,
        ["segment", "time_segment", "vehicle_type"],
        "orders_per_slot",
    )
    time_productivity_lookup = make_lookup(
        time_productivity,
        ["time_segment", "vehicle_type"],
        "orders_per_slot",
    )
    cycle_location_lookup = make_duration_lookup(
        cycle_location if cycle_location is not None else pd.DataFrame(),
        [
            "planning_location_id",
            "segment",
            "time_segment",
            "vehicle_type",
        ],
    )
    cycle_segment_lookup = make_duration_lookup(
        cycle_segment if cycle_segment is not None else pd.DataFrame(),
        ["segment", "time_segment", "vehicle_type"],
    )
    cycle_time_lookup = make_duration_lookup(
        cycle_time_segment
        if cycle_time_segment is not None
        else pd.DataFrame(),
        ["time_segment", "vehicle_type"],
    )
    cycle_global_lookup = make_duration_lookup(
        cycle_global if cycle_global is not None else pd.DataFrame(),
        ["vehicle_type"],
    )

    rows = []
    for _, row in pred_df.iterrows():
        order_shares = {
            vehicle_type: lookup_vehicle_share(
                row,
                vehicle_type,
                location_share_lookup,
                segment_share_lookup,
                global_share,
                location_overall_share_lookup,
            )
            for vehicle_type in VEHICLE_TYPES
        }
        total_share = sum(order_shares.values())
        if total_share <= 0:
            order_shares = {
                AUTO: DEFAULT_AUTO_SHARE,
                BIKE: 1 - DEFAULT_AUTO_SHARE,
            }
            total_share = 1.0
        order_shares = {k: v / total_share for k, v in order_shares.items()}
        shares = order_shares.copy()
        vehicle_share_source = "order_history"

        transport = resolve_location_transport(
            row[LOCATION_COLUMN],
            transport_by_location or {},
            group_to_members or {},
        )
        if transport == TRANSPORT_AUTO_BIKE:
            schedule_shares = (schedule_vehicle_share_lookup or {}).get(
                str(row[LOCATION_COLUMN])
            )
            if schedule_shares:
                shares = schedule_shares.copy()
                vehicle_share_source = "schedule_courier_hours"

        slot_capacity = {}
        slot_capacity_source = {}
        duration_minutes = {}
        duration_orders = {}
        duration_source = {}
        for vehicle_type in VEHICLE_TYPES:
            capacity, source = lookup_productivity(
                row,
                vehicle_type,
                location_productivity_lookup,
                segment_productivity_lookup,
                time_productivity_lookup,
                global_productivity,
            )
            slot_capacity[vehicle_type] = capacity * SLOT_CAPACITY_MULTIPLIER
            slot_capacity_source[vehicle_type] = source
            (
                duration_minutes[vehicle_type],
                duration_orders[vehicle_type],
                duration_source[vehicle_type],
            ) = lookup_cycle_time(
                row,
                vehicle_type,
                cycle_location_lookup,
                cycle_segment_lookup,
                cycle_time_lookup,
                cycle_global_lookup,
            )

        prediction = max(float(row["prediction"]), 0.0)
        (
            _,
            baseline_auto_orders,
            baseline_bike_orders,
            _,
            _,
        ) = apply_transport_to_slot_plan(
            transport,
            prediction,
            order_shares.copy(),
            slot_capacity,
        )
        (
            _,
            _,
            _,
            _,
            baseline_auto_slots,
            baseline_bike_slots,
        ) = select_slot_demand(
            baseline_auto_orders,
            baseline_bike_orders,
            slot_capacity,
            duration_minutes,
            segment_hours(row["time_segment"]),
        )
        shares, auto_orders, bike_orders, auto_slots, bike_slots = (
            apply_transport_to_slot_plan(
                transport,
                prediction,
                shares,
                slot_capacity,
            )
        )
        (
            history_raw,
            cycle_raw,
            selected_raw,
            selected_source,
            auto_slots,
            bike_slots,
        ) = select_slot_demand(
            auto_orders,
            bike_orders,
            slot_capacity,
            duration_minutes,
            segment_hours(row["time_segment"]),
        )

        rows.append({
            LOCATION_COLUMN: row[LOCATION_COLUMN],
            "segment_datetime": row["segment_datetime"],
            "segment": row["segment"],
            "time_segment": row["time_segment"],
            "orders_actual": row.get("orders_count", np.nan),
            "orders_prediction": prediction,
            "absorbed_partner_orders": float(
                row.get("absorbed_partner_orders", 0.0) or 0.0
            ),
            "absorbed_partners": str(row.get("absorbed_partners", "") or ""),
            "auto_order_prediction": auto_orders,
            "bike_order_prediction": bike_orders,
            "auto_order_share": shares[AUTO],
            "bike_order_share": shares[BIKE],
            "vehicle_share_source": vehicle_share_source,
            "auto_orders_per_slot": slot_capacity[AUTO],
            "bike_orders_per_slot": slot_capacity[BIKE],
            "auto_slot_capacity_source": slot_capacity_source[AUTO],
            "bike_slot_capacity_source": slot_capacity_source[BIKE],
            "auto_history_raw_slots": history_raw[AUTO],
            "bike_history_raw_slots": history_raw[BIKE],
            "auto_cycle_time_raw_slots": cycle_raw[AUTO],
            "bike_cycle_time_raw_slots": cycle_raw[BIKE],
            "auto_selected_raw_slots": selected_raw[AUTO],
            "bike_selected_raw_slots": selected_raw[BIKE],
            "auto_slot_demand_source": selected_source[AUTO],
            "bike_slot_demand_source": selected_source[BIKE],
            "auto_delivery_percentile_minutes": duration_minutes[AUTO],
            "bike_delivery_percentile_minutes": duration_minutes[BIKE],
            "auto_delivery_duration_orders": duration_orders[AUTO],
            "bike_delivery_duration_orders": duration_orders[BIKE],
            "auto_delivery_duration_source": duration_source[AUTO],
            "bike_delivery_duration_source": duration_source[BIKE],
            "auto_slots_needed": auto_slots,
            "bike_slots_needed": bike_slots,
            "total_slots_needed": auto_slots + bike_slots,
            "baseline_auto_slots_needed": baseline_auto_slots,
            "baseline_bike_slots_needed": baseline_bike_slots,
            "baseline_total_slots_needed": (
                baseline_auto_slots + baseline_bike_slots
            ),
            # Backward-compatible aliases for previous MVP output readers.
            "auto_orders_per_courier": slot_capacity[AUTO],
            "bike_orders_per_courier": slot_capacity[BIKE],
            "auto_couriers_needed": auto_slots,
            "bike_couriers_needed": bike_slots,
            "total_couriers_needed": auto_slots + bike_slots,
            "safety_buffer": SAFETY_BUFFER,
            "slot_capacity_multiplier": SLOT_CAPACITY_MULTIPLIER,
            "slot_rounding_mode": SLOT_ROUNDING_MODE,
            "slot_rounding_strategy": "per_segment",
        })

    return apply_joint_daily_rounding(pd.DataFrame(rows))

def build_summary(forecast_df, history_start, history_finish):
    rows = [
        {"metric": "rows", "value": len(forecast_df)},
        {"metric": "history_start", "value": str(history_start)},
        {"metric": "history_finish", "value": str(history_finish)},
        {
            "metric": "orders_prediction_sum",
            "value": forecast_df["orders_prediction"].sum(),
        },
        {
            "metric": "auto_order_prediction_sum",
            "value": forecast_df["auto_order_prediction"].sum(),
        },
        {
            "metric": "bike_order_prediction_sum",
            "value": forecast_df["bike_order_prediction"].sum(),
        },
        {
            "metric": "auto_slots_needed_sum",
            "value": forecast_df["auto_slots_needed"].sum(),
        },
        {
            "metric": "bike_slots_needed_sum",
            "value": forecast_df["bike_slots_needed"].sum(),
        },
        {
            "metric": "total_slots_needed_sum",
            "value": forecast_df["total_slots_needed"].sum(),
        },
        {"metric": "safety_buffer", "value": SAFETY_BUFFER},
        {
            "metric": "calibrated_safety_buffer",
            "value": CALIBRATED_SAFETY_BUFFER,
        },
        {"metric": "joint_daily_rounding", "value": JOINT_DAILY_ROUNDING},
        {"metric": "slot_capacity_multiplier", "value": SLOT_CAPACITY_MULTIPLIER},
        {"metric": "slot_rounding_mode", "value": SLOT_ROUNDING_MODE},
        {"metric": "cycle_time_percentile", "value": CYCLE_TIME_PERCENTILE},
        {
            "metric": "cycle_time_return_multiplier",
            "value": CYCLE_TIME_RETURN_MULTIPLIER,
        },
        {
            "metric": "default_auto_orders_per_slot_per_hour",
            "value": DEFAULT_AUTO_ORDERS_PER_SLOT_PER_HOUR,
        },
        {
            "metric": "default_bike_orders_per_slot_per_hour",
            "value": DEFAULT_BIKE_ORDERS_PER_SLOT_PER_HOUR,
        },
    ]
    return pd.DataFrame(rows)


def build_cycle_time_impact_audit(forecast_df):
    rows = []
    for row in forecast_df.itertuples(index=False):
        baseline_auto, baseline_bike = allocate_slots_from_raw_demand(
            float(row.auto_history_raw_slots),
            float(row.bike_history_raw_slots),
        )
        rows.append(
            {
                LOCATION_COLUMN: str(getattr(row, LOCATION_COLUMN)),
                "segment_datetime": row.segment_datetime,
                "segment": row.segment,
                "time_segment": row.time_segment,
                "baseline_auto_slots": baseline_auto,
                "baseline_bike_slots": baseline_bike,
                "baseline_total_slots": baseline_auto + baseline_bike,
                "selected_auto_slots": int(row.auto_slots_needed),
                "selected_bike_slots": int(row.bike_slots_needed),
                "selected_total_slots": int(row.total_slots_needed),
                "auto_slot_delta": int(row.auto_slots_needed) - baseline_auto,
                "bike_slot_delta": int(row.bike_slots_needed) - baseline_bike,
                "total_slot_delta": (
                    int(row.total_slots_needed)
                    - baseline_auto
                    - baseline_bike
                ),
                "auto_cycle_time_won": (
                    row.auto_slot_demand_source == "cycle_time"
                ),
                "bike_cycle_time_won": (
                    row.bike_slot_demand_source == "cycle_time"
                ),
                "auto_delivery_duration_source": (
                    row.auto_delivery_duration_source
                ),
                "bike_delivery_duration_source": (
                    row.bike_delivery_duration_source
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_cycle_time_impact(audit_df):
    if audit_df.empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(
        [
            {"metric": "rows", "value": len(audit_df)},
            {
                "metric": "baseline_total_slots",
                "value": int(audit_df["baseline_total_slots"].sum()),
            },
            {
                "metric": "selected_total_slots",
                "value": int(audit_df["selected_total_slots"].sum()),
            },
            {
                "metric": "total_slot_delta",
                "value": int(audit_df["total_slot_delta"].sum()),
            },
            {
                "metric": "rows_with_slot_increase",
                "value": int((audit_df["total_slot_delta"] > 0).sum()),
            },
            {
                "metric": "auto_cycle_time_win_rate",
                "value": float(audit_df["auto_cycle_time_won"].mean()),
            },
            {
                "metric": "bike_cycle_time_win_rate",
                "value": float(audit_df["bike_cycle_time_won"].mean()),
            },
            {
                "metric": "auto_missing_duration_rate",
                "value": float(
                    (
                        audit_df["auto_delivery_duration_source"]
                        == "missing_cycle_time"
                    ).mean()
                ),
            },
            {
                "metric": "bike_missing_duration_rate",
                "value": float(
                    (
                        audit_df["bike_delivery_duration_source"]
                        == "missing_cycle_time"
                    ).mean()
                ),
            },
        ]
    )


def build_capacity_model_backtest_comparison(backtest_df, impact_df):
    key_columns = [
        LOCATION_COLUMN,
        "segment_datetime",
        "segment",
        "time_segment",
    ]
    actual_columns = key_columns + [
        "actual_auto_slots",
        "actual_bike_slots",
        "actual_total_slots",
    ]
    comparison = impact_df.merge(
        backtest_df[actual_columns],
        on=key_columns,
        how="left",
    )
    rows = []
    for model in ["baseline", "selected"]:
        row = {"model": model, "rows": len(comparison)}
        for vehicle_type in ["auto", "bike", "total"]:
            forecast = comparison[f"{model}_{vehicle_type}_slots"]
            actual = comparison[f"actual_{vehicle_type}_slots"].fillna(0)
            error = forecast - actual
            actual_sum = actual.sum()
            row[f"{vehicle_type}_slot_mae"] = float(error.abs().mean())
            row[f"{vehicle_type}_under_sum"] = float(
                (-error.clip(upper=0)).sum()
            )
            row[f"{vehicle_type}_over_sum"] = float(
                error.clip(lower=0).sum()
            )
            row[f"{vehicle_type}_slot_wape"] = (
                float(error.abs().sum() / actual_sum)
                if actual_sum > 0
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_slot_backtest(forecast_df, actual_slot_df):
    key_cols = [LOCATION_COLUMN, "segment_datetime", "segment", "time_segment"]
    actual_pivot = pd.DataFrame(columns=key_cols)
    if not actual_slot_df.empty:
        actual_pivot = (
            actual_slot_df.pivot_table(
                index=key_cols,
                columns="vehicle_type",
                values="actual_slots",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reset_index()
        )
        actual_pivot.columns.name = None

    for vehicle_type in VEHICLE_TYPES:
        if vehicle_type not in actual_pivot.columns:
            actual_pivot[vehicle_type] = 0.0

    actual_pivot = actual_pivot.rename(
        columns={
            AUTO: "actual_auto_slots",
            BIKE: "actual_bike_slots",
        }
    )
    actual_pivot["actual_total_slots"] = (
        actual_pivot["actual_auto_slots"] + actual_pivot["actual_bike_slots"]
    )

    backtest = forecast_df.merge(actual_pivot, on=key_cols, how="left")
    for col in ["actual_auto_slots", "actual_bike_slots", "actual_total_slots"]:
        backtest[col] = backtest[col].fillna(0.0)

    rename_cols = {
        "auto_slots_needed": "forecast_auto_slots",
        "bike_slots_needed": "forecast_bike_slots",
        "total_slots_needed": "forecast_total_slots",
    }
    backtest = backtest.rename(columns=rename_cols)
    for vehicle_type in ["auto", "bike", "total"]:
        forecast_col = f"forecast_{vehicle_type}_slots"
        actual_col = f"actual_{vehicle_type}_slots"
        error_col = f"{vehicle_type}_slot_error"
        backtest[error_col] = backtest[forecast_col] - backtest[actual_col]
        backtest[f"{vehicle_type}_slot_abs_error"] = backtest[error_col].abs()
        backtest[f"{vehicle_type}_slot_under"] = np.where(
            backtest[error_col] < 0,
            -backtest[error_col],
            0.0,
        )
        backtest[f"{vehicle_type}_slot_over"] = np.where(
            backtest[error_col] > 0,
            backtest[error_col],
            0.0,
        )

    return backtest


def evaluate_slot_backtest(backtest_df, group_cols=None):
    if group_cols is None:
        group_cols = []

    def _metrics(group):
        row = {"rows": len(group)}
        for vehicle_type in ["auto", "bike", "total"]:
            forecast_col = f"forecast_{vehicle_type}_slots"
            actual_col = f"actual_{vehicle_type}_slots"
            error_col = f"{vehicle_type}_slot_error"
            abs_error_col = f"{vehicle_type}_slot_abs_error"
            under_col = f"{vehicle_type}_slot_under"
            over_col = f"{vehicle_type}_slot_over"
            row[f"{vehicle_type}_forecast_slots_sum"] = group[forecast_col].sum()
            row[f"{vehicle_type}_actual_slots_sum"] = group[actual_col].sum()
            row[f"{vehicle_type}_slot_bias"] = group[error_col].mean()
            row[f"{vehicle_type}_slot_mae"] = group[abs_error_col].mean()
            row[f"{vehicle_type}_slot_rmse"] = np.sqrt((group[error_col] ** 2).mean())
            row[f"{vehicle_type}_slot_under_sum"] = group[under_col].sum()
            row[f"{vehicle_type}_slot_over_sum"] = group[over_col].sum()
            row[f"{vehicle_type}_under_rate"] = (group[error_col] < 0).mean()
            actual_sum = group[actual_col].sum()
            row[f"{vehicle_type}_slot_wape"] = (
                group[abs_error_col].sum() / actual_sum if actual_sum > 0 else np.nan
            )
        return pd.Series(row)

    if not group_cols:
        return pd.DataFrame([_metrics(backtest_df)])
    return backtest_df.groupby(group_cols).apply(_metrics).reset_index()


def build_simple_slot_comparison(backtest_df):
    columns = [
        LOCATION_COLUMN,
        "segment_datetime",
        "segment",
        "time_segment",
        "actual_orders",
        "predicted_orders",
        "forecast_auto_slots",
        "forecast_bike_slots",
        "forecast_total_slots",
        "actual_auto_slots",
        "actual_bike_slots",
        "actual_total_slots",
    ]
    result_df = backtest_df.rename(
        columns={
            "orders_actual": "actual_orders",
            "orders_prediction": "predicted_orders",
        }
    )
    return result_df[columns].sort_values(
        [LOCATION_COLUMN, "segment_datetime", "time_segment"]
    )


def main():
    partner_grouping = load_partner_grouping()
    location_to_group = partner_grouping.location_to_planning
    group_to_members = partner_grouping.planning_to_members
    pred_df = load_predictions(PREDICTIONS_PATH)
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows found in {PREDICTIONS_PATH}")
    pred_df, partner_order_audit_df = aggregate_predictions_to_planning_locations(
        pred_df,
        partner_grouping,
    )

    location_segment_map = build_location_segment_map(pred_df)
    history_finish = pred_df["segment_datetime"].min()
    history_start = history_finish - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)
    history_start_ms = datetime_to_ms(history_start)
    history_finish_ms = datetime_to_ms(history_finish)
    history_segment_grid = build_historical_segment_grid(
        location_segment_map,
        history_start,
        history_finish,
    )

    engine = build_engine()
    transport_by_location = load_location_transport(engine)
    order_df = load_order_vehicle_history(engine, history_start_ms, history_finish_ms)
    order_df = prepare_order_history(order_df, location_to_group, location_segment_map)
    location_overall_share_lookup = build_location_overall_share_lookup(order_df)
    order_counts, location_share, segment_share, global_share = build_order_share_tables(
        order_df
    )

    schedule_df = load_schedule_history(engine, history_start_ms, history_finish_ms)
    schedule_df = prepare_schedule_history(schedule_df, location_to_group)
    schedule_vehicle_share_lookup = build_schedule_vehicle_share_lookup(schedule_df)
    courier_equiv = build_courier_equivalent_by_segment(
        schedule_df,
        history_segment_grid,
    )
    (
        location_productivity,
        segment_productivity,
        time_productivity,
        global_productivity,
    ) = build_productivity_tables(order_counts, courier_equiv)
    (
        cycle_location,
        cycle_segment,
        cycle_time_segment,
        cycle_global,
    ) = load_cycle_time_tables()

    forecast_df = build_courier_forecast(
        pred_df,
        location_share,
        segment_share,
        global_share,
        location_productivity,
        segment_productivity,
        time_productivity,
        global_productivity,
        transport_by_location,
        group_to_members,
        location_overall_share_lookup,
        schedule_vehicle_share_lookup,
        cycle_location,
        cycle_segment,
        cycle_time_segment,
        cycle_global,
    )
    prediction_start = pred_df["segment_datetime"].min()
    prediction_finish = pred_df["segment_end"].max()
    prediction_schedule_df = load_schedule_history(
        engine,
        datetime_to_ms(prediction_start),
        datetime_to_ms(prediction_finish),
    )
    prediction_schedule_df = prepare_schedule_history(
        prediction_schedule_df,
        location_to_group,
    )
    actual_slot_df = build_slot_equivalent_by_window(
        prediction_schedule_df,
        pred_df,
    )
    backtest_df = build_slot_backtest(forecast_df, actual_slot_df)
    slot_metrics_df = evaluate_slot_backtest(backtest_df)
    slot_metrics_by_segment_df = evaluate_slot_backtest(backtest_df, ["segment"])
    slot_metrics_by_segment_time_df = evaluate_slot_backtest(
        backtest_df,
        ["segment", "time_segment"],
    )
    slot_metrics_by_location_df = evaluate_slot_backtest(
        backtest_df,
        [LOCATION_COLUMN, "segment"],
    )
    simple_slot_comparison_df = build_simple_slot_comparison(backtest_df)
    summary_df = build_summary(forecast_df, history_start, history_finish)
    cycle_impact_df = build_cycle_time_impact_audit(forecast_df)
    cycle_impact_summary_df = summarize_cycle_time_impact(cycle_impact_df)
    capacity_model_comparison_df = build_capacity_model_backtest_comparison(
        backtest_df,
        cycle_impact_df,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_DIR / "courier_forecast.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "courier_forecast_summary.csv", index=False)
    cycle_impact_df.to_csv(
        OUTPUT_DIR / "cycle_time_impact_audit.csv",
        index=False,
    )
    cycle_impact_summary_df.to_csv(
        OUTPUT_DIR / "cycle_time_impact_summary.csv",
        index=False,
    )
    capacity_model_comparison_df.to_csv(
        OUTPUT_DIR / "capacity_model_backtest_comparison.csv",
        index=False,
    )
    location_share.to_csv(OUTPUT_DIR / "vehicle_share_by_location.csv", index=False)
    segment_share.to_csv(OUTPUT_DIR / "vehicle_share_by_segment.csv", index=False)
    location_productivity.to_csv(
        OUTPUT_DIR / "courier_productivity_by_location.csv",
        index=False,
    )
    segment_productivity.to_csv(
        OUTPUT_DIR / "courier_productivity_by_segment.csv",
        index=False,
    )
    time_productivity.to_csv(
        OUTPUT_DIR / "courier_productivity_by_time_segment.csv",
        index=False,
    )
    courier_equiv.to_csv(OUTPUT_DIR / "courier_equivalent_capacity.csv", index=False)
    for name, table in {
        "location": cycle_location,
        "segment": cycle_segment,
        "time_segment": cycle_time_segment,
        "global": cycle_global,
    }.items():
        table.to_csv(
            OUTPUT_DIR / f"courier_cycle_time_capacity_{name}.csv",
            index=False,
        )
    history_segment_grid.to_csv(OUTPUT_DIR / "historical_segment_grid.csv", index=False)
    actual_slot_df.to_csv(OUTPUT_DIR / "actual_slots_by_window.csv", index=False)
    backtest_df.to_csv(OUTPUT_DIR / "courier_slot_backtest.csv", index=False)
    simple_slot_comparison_df.to_csv(
        OUTPUT_DIR / "courier_slot_comparison_simple.csv",
        index=False,
    )
    slot_metrics_df.to_csv(OUTPUT_DIR / "courier_slot_metrics.csv", index=False)
    slot_metrics_by_segment_df.to_csv(
        OUTPUT_DIR / "courier_slot_metrics_by_segment.csv",
        index=False,
    )
    slot_metrics_by_segment_time_df.to_csv(
        OUTPUT_DIR / "courier_slot_metrics_by_segment_time.csv",
        index=False,
    )
    slot_metrics_by_location_df.to_csv(
        OUTPUT_DIR / "courier_slot_metrics_by_location.csv",
        index=False,
    )
    partner_order_audit_df.to_csv(
        OUTPUT_DIR / "partner_grouping_order_audit.csv",
        index=False,
    )
    build_grouping_config_audit(partner_grouping).to_csv(
        OUTPUT_DIR / "partner_grouping_config_audit.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "planning_location_id": planning_id,
                "member_location_ids": ",".join(members),
            }
            for planning_id, members in sorted(group_to_members.items())
        ]
    ).to_csv(
        OUTPUT_DIR / "planning_location_groups.csv",
        index=False,
    )

    print(f"Saved courier forecast to: {OUTPUT_DIR}")
    print(summary_df.to_string(index=False))
    print("\nSlot backtest:")
    print(slot_metrics_df.to_string(index=False))
    print("\nGlobal vehicle share:", global_share)
    print("Global slot capacity from shifts:", global_productivity)
    print(
        "Default slot capacity per hour:",
        {
            AUTO: DEFAULT_AUTO_ORDERS_PER_SLOT_PER_HOUR,
            BIKE: DEFAULT_BIKE_ORDERS_PER_SLOT_PER_HOUR,
        },
    )


if __name__ == "__main__":
    main()
