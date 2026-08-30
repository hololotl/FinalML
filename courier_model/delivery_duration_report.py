import os
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from . import main as cm
    from .partner_grouping import load_partner_grouping, normalize_location_id
except ImportError:
    import main as cm
    from partner_grouping import load_partner_grouping, normalize_location_id


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = Path(
    os.getenv(
        "DELIVERY_DURATION_OUTPUT_DIR",
        SCRIPT_DIR / "res" / "delivery_duration",
    )
)
SEGMENT_SOURCE_PATH = Path(
    os.getenv(
        "DELIVERY_DURATION_SEGMENT_SOURCE",
        PROJECT_DIR / "week_model" / "res" / "predictions_full.csv",
    )
)
HISTORY_START = os.getenv("DELIVERY_DURATION_HISTORY_START", "").strip()
HISTORY_FINISH = os.getenv("DELIVERY_DURATION_HISTORY_FINISH", "").strip()
MAX_DURATION_MINUTES = float(
    os.getenv("DELIVERY_DURATION_MAX_MINUTES", "360")
)
MIN_DURATION_MINUTES = float(
    os.getenv("DELIVERY_DURATION_MIN_MINUTES", "0")
)

TIME_SEGMENT_WINDOWS = {
    "block_00_06": "00:00-06:00",
    "block_06_12": "06:00-12:00",
    "block_12_18": "12:00-18:00",
    "block_18_24": "18:00-24:00",
    "deep_night": "00:00-06:00",
    "opening": "06:00-08:00",
    "early_morning": "08:00-09:00",
    "morning": "09:00-11:00",
    "lunch": "11:00-15:00",
    "afternoon": "15:00-18:00",
    "dinner": "18:00-21:00",
    "late_evening": "21:00-24:00",
}


def parse_datetime_to_ms(value):
    if not value:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Europe/Moscow")
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


def load_order_durations(
    engine,
    history_start=HISTORY_START,
    history_finish=HISTORY_FINISH,
):
    conditions = [
        "delivering_at IS NOT NULL",
        "finished_at IS NOT NULL",
    ]
    params = {}
    history_start_ms = parse_datetime_to_ms(history_start)
    history_finish_ms = parse_datetime_to_ms(history_finish)
    if history_start_ms is not None:
        conditions.append("delivering_at >= :history_start_ms")
        params["history_start_ms"] = history_start_ms
    if history_finish_ms is not None:
        conditions.append("delivering_at < :history_finish_ms")
        params["history_finish_ms"] = history_finish_ms

    query = text(
        """
        SELECT
            o.id AS order_id,
            o.location_id,
            o.delivering_at,
            o.finished_at,
            COALESCE(c.vehicle, 2) AS vehicle
        FROM orders o
        LEFT JOIN couriers c ON o.courier_id = c.id
        WHERE
        """
        + " AND ".join(conditions)
    )
    return pd.read_sql_query(query, engine, params=params)


def load_location_names(engine):
    locations = pd.read_sql_query(
        text("SELECT id AS location_id, name AS location_name FROM locations"),
        engine,
    )
    locations["location_id"] = locations["location_id"].map(normalize_location_id)
    return dict(zip(locations["location_id"], locations["location_name"]))


def load_location_segment_map(path):
    if not path.exists():
        raise FileNotFoundError(
            "Location segment source not found. Train week_model first: "
            f"{path}"
        )
    segments = pd.read_csv(path, dtype={cm.LOCATION_COLUMN: str})
    required = {cm.LOCATION_COLUMN, "segment"}
    missing = sorted(required - set(segments.columns))
    if missing:
        raise ValueError(f"Segment source is missing columns: {missing}")
    segments[cm.LOCATION_COLUMN] = segments[cm.LOCATION_COLUMN].map(
        normalize_location_id
    )
    return (
        segments[[cm.LOCATION_COLUMN, "segment"]]
        .dropna()
        .drop_duplicates(subset=[cm.LOCATION_COLUMN])
        .set_index(cm.LOCATION_COLUMN)["segment"]
        .to_dict()
    )


def prepare_duration_rows(order_df, grouping, segment_map, location_names):
    rows = order_df.copy()
    rows["source_location_id"] = rows["location_id"].map(normalize_location_id)
    rows["planning_location_id"] = rows["source_location_id"].map(
        grouping.planning_location_id
    )
    rows["location_name"] = rows["source_location_id"].map(location_names)
    rows["planning_location_name"] = rows["planning_location_id"].map(
        location_names
    )
    rows["segment"] = rows["planning_location_id"].map(segment_map)
    rows["vehicle_type"] = rows["vehicle"].map(cm.vehicle_to_type)
    rows["segment_source"] = np.where(
        rows["segment"].notna(),
        "week_model",
        "historical_volume_fallback",
    )
    planning_counts = rows.groupby("planning_location_id").size()
    p50 = planning_counts.quantile(0.50)
    p85 = planning_counts.quantile(0.85)
    p95 = planning_counts.quantile(0.95)

    def fallback_segment(planning_location_id):
        count = planning_counts.get(planning_location_id, 0)
        if count <= p50:
            return "low"
        if count <= p85:
            return "medium"
        if count <= p95:
            return "high"
        return "mega"

    missing_segment = rows["segment"].isna()
    rows.loc[missing_segment, "segment"] = rows.loc[
        missing_segment,
        "planning_location_id",
    ].map(fallback_segment)

    rows["delivering_datetime"] = cm.to_moscow_datetime(rows["delivering_at"])
    rows["finished_datetime"] = cm.to_moscow_datetime(rows["finished_at"])
    rows["duration_minutes"] = (
        rows["finished_datetime"] - rows["delivering_datetime"]
    ).dt.total_seconds() / 60.0
    rows["delivery_hour"] = rows["delivering_datetime"].dt.hour
    rows["time_segment"] = rows.apply(
        lambda row: (
            cm.classify_time_segment(row["delivery_hour"], row["segment"])
            if pd.notna(row["segment"])
            else None
        ),
        axis=1,
    )

    rows["valid_timestamp_order"] = rows["duration_minutes"] > 0
    rows["inside_duration_limits"] = rows["duration_minutes"].between(
        MIN_DURATION_MINUTES,
        MAX_DURATION_MINUTES,
        inclusive="right",
    )
    rows["included_in_report"] = (
        rows["valid_timestamp_order"]
        & rows["inside_duration_limits"]
        & rows["segment"].notna()
        & rows["time_segment"].notna()
    )
    return rows


def summarize_duration(group):
    durations = group["duration_minutes"]
    return pd.Series(
        {
            "orders_count": int(durations.size),
            "average_minutes": durations.mean(),
            "median_minutes": durations.median(),
            "p70_minutes": durations.quantile(0.70),
            "p80_minutes": durations.quantile(0.80),
            "p90_minutes": durations.quantile(0.90),
            "min_minutes": durations.min(),
            "max_minutes": durations.max(),
            "std_minutes": durations.std(),
        }
    )


def build_time_segment_report(valid_rows):
    group_columns = [
        "source_location_id",
        "location_name",
        "planning_location_id",
        "planning_location_name",
        "segment",
        "segment_source",
        "time_segment",
        "vehicle_type",
    ]
    report = (
        valid_rows.groupby(group_columns, dropna=False)
        .apply(summarize_duration, include_groups=False)
        .reset_index()
    )
    report["time_window"] = report["time_segment"].map(TIME_SEGMENT_WINDOWS)
    column_order = group_columns + [
        "time_window",
        "orders_count",
        "average_minutes",
        "median_minutes",
        "p70_minutes",
        "p80_minutes",
        "p90_minutes",
        "min_minutes",
        "max_minutes",
        "std_minutes",
    ]
    return report[column_order].sort_values(
        ["source_location_id", "time_window"]
    )


def build_hourly_report(valid_rows):
    group_columns = [
        "source_location_id",
        "location_name",
        "planning_location_id",
        "planning_location_name",
        "segment",
        "segment_source",
        "time_segment",
        "vehicle_type",
        "delivery_hour",
    ]
    report = (
        valid_rows.groupby(group_columns, dropna=False)
        .apply(summarize_duration, include_groups=False)
        .reset_index()
    )
    report["hour_window"] = report["delivery_hour"].map(
        lambda hour: f"{int(hour):02d}:00-{int(hour) + 1:02d}:00"
    )
    column_order = group_columns + [
        "hour_window",
        "orders_count",
        "average_minutes",
        "median_minutes",
        "p70_minutes",
        "p80_minutes",
        "p90_minutes",
        "min_minutes",
        "max_minutes",
        "std_minutes",
    ]
    return report[column_order].sort_values(
        ["source_location_id", "delivery_hour"]
    )


def build_planning_time_segment_report(valid_rows):
    group_columns = [
        "planning_location_id",
        "planning_location_name",
        "segment",
        "time_segment",
        "vehicle_type",
    ]
    report = (
        valid_rows.groupby(group_columns, dropna=False)
        .apply(summarize_duration, include_groups=False)
        .reset_index()
    )
    report["time_window"] = report["time_segment"].map(TIME_SEGMENT_WINDOWS)
    return report[
        group_columns
        + [
            "time_window",
            "orders_count",
            "average_minutes",
            "median_minutes",
            "p70_minutes",
            "p80_minutes",
            "p90_minutes",
            "min_minutes",
            "max_minutes",
            "std_minutes",
        ]
    ].sort_values(["planning_location_id", "time_window", "vehicle_type"])


def build_planning_hourly_report(valid_rows):
    group_columns = [
        "planning_location_id",
        "planning_location_name",
        "segment",
        "time_segment",
        "vehicle_type",
        "delivery_hour",
    ]
    report = (
        valid_rows.groupby(group_columns, dropna=False)
        .apply(summarize_duration, include_groups=False)
        .reset_index()
    )
    report["hour_window"] = report["delivery_hour"].map(
        lambda hour: f"{int(hour):02d}:00-{int(hour) + 1:02d}:00"
    )
    return report[
        group_columns
        + [
            "hour_window",
            "orders_count",
            "average_minutes",
            "median_minutes",
            "p70_minutes",
            "p80_minutes",
            "p90_minutes",
            "min_minutes",
            "max_minutes",
            "std_minutes",
        ]
    ].sort_values(["planning_location_id", "delivery_hour", "vehicle_type"])


def build_quality_summary(
    prepared_rows,
    history_start=HISTORY_START,
    history_finish=HISTORY_FINISH,
):
    return pd.DataFrame(
        [
            {
                "metric": "orders_with_both_timestamps",
                "value": len(prepared_rows),
            },
            {
                "metric": "orders_included",
                "value": int(prepared_rows["included_in_report"].sum()),
            },
            {
                "metric": "invalid_or_nonpositive_duration",
                "value": int((~prepared_rows["valid_timestamp_order"]).sum()),
            },
            {
                "metric": "outside_duration_limits",
                "value": int(
                    (
                        prepared_rows["valid_timestamp_order"]
                        & ~prepared_rows["inside_duration_limits"]
                    ).sum()
                ),
            },
            {
                "metric": "orders_without_location_segment",
                "value": int(prepared_rows["segment"].isna().sum()),
            },
            {
                "metric": "min_duration_minutes",
                "value": MIN_DURATION_MINUTES,
            },
            {
                "metric": "max_duration_minutes",
                "value": MAX_DURATION_MINUTES,
            },
            {
                "metric": "history_start",
                "value": history_start or "all",
            },
            {
                "metric": "history_finish",
                "value": history_finish or "all",
            },
        ]
    )


def round_duration_columns(report):
    result = report.copy()
    minute_columns = [
        column
        for column in result.columns
        if column.endswith("_minutes")
    ]
    result[minute_columns] = result[minute_columns].round(2)
    result["orders_count"] = result["orders_count"].astype(int)
    return result


def main():
    engine = create_engine(cm.DB_URL)
    grouping = load_partner_grouping()
    segment_map = load_location_segment_map(SEGMENT_SOURCE_PATH)
    location_names = load_location_names(engine)
    history_start = HISTORY_START
    history_finish = HISTORY_FINISH
    if not history_finish:
        prediction_dates = pd.read_csv(
            SEGMENT_SOURCE_PATH,
            usecols=["segment_datetime"],
        )
        forecast_start = pd.to_datetime(
            prediction_dates["segment_datetime"],
            errors="coerce",
            utc=True,
        ).min()
        if pd.isna(forecast_start):
            raise ValueError("Cannot infer forecast start for duration history.")
        history_finish = str(forecast_start.tz_convert("Europe/Moscow"))
        history_start = history_start or str(
            forecast_start.tz_convert("Europe/Moscow") - pd.Timedelta(days=30)
        )
    order_df = load_order_durations(engine, history_start, history_finish)
    prepared = prepare_duration_rows(
        order_df,
        grouping,
        segment_map,
        location_names,
    )
    valid_rows = prepared.loc[prepared["included_in_report"]].copy()
    if valid_rows.empty:
        raise RuntimeError("No valid delivering_at → finished_at durations found.")

    segment_report = round_duration_columns(
        build_time_segment_report(valid_rows)
    )
    hourly_report = round_duration_columns(build_hourly_report(valid_rows))
    planning_segment_report = round_duration_columns(
        build_planning_time_segment_report(valid_rows)
    )
    planning_hourly_report = round_duration_columns(
        build_planning_hourly_report(valid_rows)
    )
    quality_summary = build_quality_summary(
        prepared,
        history_start,
        history_finish,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segment_report.to_csv(
        OUTPUT_DIR / "delivery_duration_by_time_segment.csv",
        index=False,
    )
    hourly_report.to_csv(
        OUTPUT_DIR / "delivery_duration_by_hour.csv",
        index=False,
    )
    planning_segment_report.to_csv(
        OUTPUT_DIR / "delivery_duration_by_planning_time_segment.csv",
        index=False,
    )
    planning_hourly_report.to_csv(
        OUTPUT_DIR / "delivery_duration_by_planning_hour.csv",
        index=False,
    )
    quality_summary.to_csv(
        OUTPUT_DIR / "delivery_duration_data_quality.csv",
        index=False,
    )

    print(f"Saved delivery duration reports to: {OUTPUT_DIR}")
    print(quality_summary.to_string(index=False))


if __name__ == "__main__":
    main()
