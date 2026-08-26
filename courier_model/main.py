import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


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

AUTO = "auto"
BIKE = "bike"
VEHICLE_TYPES = [AUTO, BIKE]

TRANSPORT_AUTO_BIKE = 0
TRANSPORT_AUTO = 1
TRANSPORT_BIKE = 2

MERGED_LOCATION_PAIRS = [
    ("19", "54", 387),
    ("71", "94", 200),
    ("102", "125", 187),
    ("288", "424", 150),
    ("62", "140", 142),
    ("136", "292", 134),
    ("187", "296", 89),
    ("130", "139", 62),
    ("122", "131", 32),
    ("98", "298", 22),
    ("336", "340", 12),
    ("302", "443", 4),
    ("343", "346", 2),
    ("420", "481", 2),
    ("341", "347", 2),
]


def build_engine():
    return create_engine(DB_URL)


def normalize_location_id(value):
    value_str = str(value).strip()
    if value_str.endswith(".0"):
        candidate = value_str[:-2]
        if candidate.isdigit():
            return candidate
    return value_str


def build_merged_location_groups():
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for left_id, right_id, _ in MERGED_LOCATION_PAIRS:
        union(normalize_location_id(left_id), normalize_location_id(right_id))

    members_by_root = {}
    for left_id, right_id, _ in MERGED_LOCATION_PAIRS:
        for location_id in [
            normalize_location_id(left_id),
            normalize_location_id(right_id),
        ]:
            root = find(location_id)
            members_by_root.setdefault(root, set()).add(location_id)

    location_to_group = {}
    for members in members_by_root.values():
        sorted_members = sorted(members, key=int)
        group_id = "grp_" + "_".join(sorted_members)
        for member in sorted_members:
            location_to_group[member] = group_id

    return location_to_group


def build_group_to_members(location_to_group):
    group_to_members = {}
    for member_id, group_id in location_to_group.items():
        group_to_members.setdefault(str(group_id), []).append(str(member_id))
    return group_to_members


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


def make_lookup(df, key_cols, value_col):
    if df.empty:
        return {}
    return {
        tuple(row[col] for col in key_cols): float(row[value_col])
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
    """When a location always runs only auto or only bike, lock the split."""
    if schedule_df.empty:
        return {}

    counts = (
        schedule_df.groupby([schedule_df[LOCATION_COLUMN].astype(str), "vehicle_type"])
        .size()
        .reset_index(name="shifts")
    )
    lookup = {}
    for location_id, group in counts.groupby(LOCATION_COLUMN):
        auto_shifts = int(group.loc[group["vehicle_type"] == AUTO, "shifts"].sum())
        bike_shifts = int(group.loc[group["vehicle_type"] == BIKE, "shifts"].sum())
        if bike_shifts == 0 and auto_shifts > 0:
            lookup[str(location_id)] = {AUTO: 1.0, BIKE: 0.0}
        elif auto_shifts == 0 and bike_shifts > 0:
            lookup[str(location_id)] = {AUTO: 0.0, BIKE: 1.0}
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

    rows = []
    for _, row in pred_df.iterrows():
        shares = {
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
        total_share = sum(shares.values())
        if total_share <= 0:
            shares = {AUTO: DEFAULT_AUTO_SHARE, BIKE: 1 - DEFAULT_AUTO_SHARE}
            total_share = 1.0
        shares = {k: v / total_share for k, v in shares.items()}

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

        slot_capacity = {}
        slot_capacity_source = {}
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

        prediction = max(float(row["prediction"]), 0.0)
        shares, auto_orders, bike_orders, auto_slots, bike_slots = (
            apply_transport_to_slot_plan(
                transport,
                prediction,
                shares,
                slot_capacity,
            )
        )

        rows.append({
            LOCATION_COLUMN: row[LOCATION_COLUMN],
            "segment_datetime": row["segment_datetime"],
            "segment": row["segment"],
            "time_segment": row["time_segment"],
            "orders_actual": row.get("orders_count", np.nan),
            "orders_prediction": prediction,
            "auto_order_prediction": auto_orders,
            "bike_order_prediction": bike_orders,
            "auto_order_share": shares[AUTO],
            "bike_order_share": shares[BIKE],
            "auto_orders_per_slot": slot_capacity[AUTO],
            "bike_orders_per_slot": slot_capacity[BIKE],
            "auto_slot_capacity_source": slot_capacity_source[AUTO],
            "bike_slot_capacity_source": slot_capacity_source[BIKE],
            "auto_slots_needed": auto_slots,
            "bike_slots_needed": bike_slots,
            "total_slots_needed": auto_slots + bike_slots,
            # Backward-compatible aliases for previous MVP output readers.
            "auto_orders_per_courier": slot_capacity[AUTO],
            "bike_orders_per_courier": slot_capacity[BIKE],
            "auto_couriers_needed": auto_slots,
            "bike_couriers_needed": bike_slots,
            "total_couriers_needed": auto_slots + bike_slots,
            "safety_buffer": SAFETY_BUFFER,
            "slot_capacity_multiplier": SLOT_CAPACITY_MULTIPLIER,
            "slot_rounding_mode": SLOT_ROUNDING_MODE,
        })

    return pd.DataFrame(rows)

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
        {"metric": "slot_capacity_multiplier", "value": SLOT_CAPACITY_MULTIPLIER},
        {"metric": "slot_rounding_mode", "value": SLOT_ROUNDING_MODE},
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
    location_to_group = build_merged_location_groups()
    pred_df = load_predictions(PREDICTIONS_PATH)
    if pred_df.empty:
        raise RuntimeError(f"No prediction rows found in {PREDICTIONS_PATH}")

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
    group_to_members = build_group_to_members(location_to_group)
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(OUTPUT_DIR / "courier_forecast.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "courier_forecast_summary.csv", index=False)
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
