import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

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
HISTORY_LOOKBACK_DAYS = int(os.getenv("COURIER_HISTORY_LOOKBACK_DAYS", "14"))

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://courier:1337@localhost:1338/coffee",
)

def build_engine():
    return create_engine(DB_URL)

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


def build_merged_location_groups():
    parent = {}

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

pd.set_option("display.max_columns", None)

def build_location_segment_map(pred_df):
    return (
        pred_df[[LOCATION_COLUMN, "segment"]]
        .dropna()
        .drop_duplicates(subset=[LOCATION_COLUMN])
        .set_index(LOCATION_COLUMN)["segment"]
        .to_dict()
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

def build_historical_segment_grid(location_segment_map, history_start, history_finish):
    rows = []
    start_day = pd.Timestamp(history_start).floor("D") # начало и конец
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

def datetime_to_ms(dt_value):
    ts = pd.Timestamp(dt_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Moscow")
    return int(ts.tz_convert("UTC").timestamp() * 1000)

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

def main():
    pred_df = load_predictions(PREDICTIONS_PATH)

    # получаем словарь, локация - ее группа (mega, low)
    location_segment_map = build_location_segment_map(pred_df)

    # считаем конец и начало, которое мы будем анализировать
    history_finish = pred_df["segment_datetime"].min()
    history_start = history_finish - pd.Timedelta(days=HISTORY_LOOKBACK_DAYS)
    history_start_ms = datetime_to_ms(history_start)
    history_finish_ms = datetime_to_ms(history_finish)

    # делаем словарь, добавляем некоторые столбцы, в location_segment_map, получаем сетку из 14 дней, где
    # расписанна каждая локация, для нее на все 14 дней сегменты временные, получим что то типо такого
    # {"location_id": "101", "segment": "high", "time_segment": "lunch", ...},
    history_segment_grid = build_historical_segment_grid(
        location_segment_map,
        history_start,
        history_finish,
    )

    engine = build_engine()
    # получаем в каком формате работают локации, 1 - auto, 2 - bike, 0 - auto and bike
    transport_by_location = load_location_transport(engine)

    ## загружаем заказы за последние 2 недели
    order_df = load_order_vehicle_history(engine, history_start_ms, history_finish_ms)



if __name__ == "__main__":
    main()


