import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import main as wm


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("WEEK_FORECAST_OUTPUT_DIR", SCRIPT_DIR / "res"))
MODEL_ARTIFACT_PATH = Path(
    os.getenv(
        "WEEK_MODEL_ARTIFACT_PATH",
        OUTPUT_DIR / "week_model_artifact.pkl",
    )
)
FORECAST_START = os.getenv("FORECAST_START", "2026-06-26 00:00:00+03:00")
FORECAST_FINISH = os.getenv("FORECAST_FINISH", "2026-07-03 00:00:00+03:00")
OUTPUT_PATH = Path(
    os.getenv(
        "WEEK_FORECAST_OUTPUT_PATH",
        OUTPUT_DIR / "future_predictions.csv",
    )
)
FORECAST_ORDERS_LIMIT = int(os.getenv("FORECAST_ORDERS_LIMIT", "2000000"))


_START_TIME = time.perf_counter()


def log_step(message, df=None):
    elapsed = time.perf_counter() - _START_TIME
    suffix = ""
    if df is not None:
        suffix = f" rows={len(df):,} cols={len(df.columns):,}"
    print(f"[forecast +{elapsed:8.1f}s] {message}{suffix}", flush=True)


def parse_moscow_datetime(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("Europe/Moscow")
    return ts.tz_convert("Europe/Moscow")


def load_model_artifact(path):
    log_step(f"Loading model artifact: {path}")
    with open(path, "rb") as file:
        artifact = pickle.load(file)
    log_step("Loaded model artifact")
    return artifact


def expand_segment_grid_until(segment_df, finish_datetime):
    log_step("Expanding segment grid until forecast finish", segment_df)
    grids = []
    finish_day = pd.Timestamp(finish_datetime).floor("D")
    for location_id, group in segment_df.groupby(wm.LOCATION_COLUMN):
        location_segment = group["segment"].iloc[0]
        segment_order = wm.get_segment_order_for_rest_segment(location_segment)
        segments_per_day = len(segment_order)
        segments = pd.DataFrame({
            "time_segment": segment_order,
            "segment_rank": np.arange(segments_per_day, dtype=int),
            "segments_per_day": segments_per_day,
        })
        date_index = pd.date_range(
            group["date"].min(),
            finish_day,
            freq="D",
        )
        grid = pd.MultiIndex.from_product(
            [date_index, segments["time_segment"]],
            names=["date", "time_segment"],
        ).to_frame(index=False)
        grid = grid.merge(segments, on="time_segment", how="left")
        grid[wm.LOCATION_COLUMN] = location_id
        grid["segment"] = location_segment
        grids.append(grid)

    grid_df = pd.concat(grids, ignore_index=True)
    merged = grid_df.merge(
        segment_df,
        on=[
            wm.LOCATION_COLUMN,
            "segment",
            "date",
            "time_segment",
            "segment_rank",
        ],
        how="left",
    )
    merged["segments_per_day"] = merged["segments_per_day"].fillna(8).astype(int)
    merged["segment_hours_step"] = 24 / merged["segments_per_day"]
    merged["segment_datetime"] = (
        pd.to_datetime(merged["date"])
        + pd.to_timedelta(
            merged["segment_rank"] * merged["segment_hours_step"],
            unit="h",
        )
    )
    merged["weekday"] = pd.to_datetime(merged["date"]).dt.dayofweek
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    merged["is_weekend"] = merged["weekday"].isin([5, 6]).astype(int)
    merged["is_holiday"] = list(
        zip(merged["month"], pd.to_datetime(merged["date"]).dt.day)
    )
    merged["is_holiday"] = (
        pd.Series(merged["is_holiday"], index=merged.index)
        .isin(wm.RUSSIAN_FIXED_HOLIDAYS)
        .astype(int)
    )
    merged = wm.add_calendar_effect_features(merged, "date")
    log_step("Expanded segment grid", merged)
    return merged


def add_future_open_hours(segment_grid, historical_segment_df, forecast_start):
    log_step("Adding future open_hours fallback values", segment_grid)
    segment_grid = segment_grid.copy()
    history = historical_segment_df.copy()
    history["weekday"] = pd.to_datetime(history["date"]).dt.dayofweek

    loc_weekday_time = (
        history.groupby([wm.LOCATION_COLUMN, "weekday", "time_segment"])["open_hours"]
        .mean()
        .to_dict()
    )
    loc_time = (
        history.groupby([wm.LOCATION_COLUMN, "time_segment"])["open_hours"]
        .mean()
        .to_dict()
    )
    segment_time = (
        history.groupby(["segment", "time_segment"])["open_hours"]
        .mean()
        .to_dict()
    )
    time_only = history.groupby("time_segment")["open_hours"].mean().to_dict()

    is_future = segment_grid["segment_datetime"] >= forecast_start
    future_rows = segment_grid.loc[is_future]
    future_open_hours = []
    for row in future_rows.itertuples(index=False):
        location_id = getattr(row, wm.LOCATION_COLUMN)
        key = (location_id, row.weekday, row.time_segment)
        fallback_key = (location_id, row.time_segment)
        segment_key = (row.segment, row.time_segment)
        open_hours = loc_weekday_time.get(key)
        if pd.isna(open_hours):
            open_hours = loc_time.get(fallback_key)
        if pd.isna(open_hours):
            open_hours = segment_time.get(segment_key)
        if pd.isna(open_hours):
            open_hours = time_only.get(row.time_segment)
        if pd.isna(open_hours):
            open_hours = row.segment_hours_step
        future_open_hours.append(float(open_hours))

    segment_grid.loc[is_future, "open_hours"] = future_open_hours
    segment_grid.loc[is_future, "orders_count"] = np.nan
    segment_grid["open_hours"] = segment_grid["open_hours"].fillna(0)
    log_step("Added future open_hours", segment_grid)
    return segment_grid


def add_is_open_flag_fast(df, work_hours_df):
    if work_hours_df is None or work_hours_df.empty:
        df = df.copy()
        df["is_open"] = True
        return df

    work_hours = work_hours_df[
        [wm.LOCATION_COLUMN, "weekday", "start_hour", "finish_hour"]
    ].copy()
    work_hours[wm.LOCATION_COLUMN] = work_hours[wm.LOCATION_COLUMN].astype(str)
    work_hours = work_hours.drop_duplicates(
        subset=[wm.LOCATION_COLUMN, "weekday"],
        keep="first",
    )

    result = df.merge(
        work_hours,
        on=[wm.LOCATION_COLUMN, "weekday"],
        how="left",
    )
    has_work_hours = result["start_hour"].notna() & result["finish_hour"].notna()
    same_day = result["start_hour"] < result["finish_hour"]
    overnight = has_work_hours & ~same_day

    result["is_open"] = False
    result.loc[
        has_work_hours
        & same_day
        & (result["hour"] >= result["start_hour"])
        & (result["hour"] < result["finish_hour"]),
        "is_open",
    ] = True
    result.loc[
        overnight
        & (
            (result["hour"] >= result["start_hour"])
            | (result["hour"] < result["finish_hour"])
        ),
        "is_open",
    ] = True

    return result.drop(columns=["start_hour", "finish_hour"])


def add_time_segment_fast(df, location_segment_map, location_to_group):
    result = df.copy()
    location_key = (
        result[wm.LOCATION_COLUMN]
        .map(wm.normalize_location_id)
        .map(lambda x: location_to_group.get(x, x))
    )
    restaurant_segment = location_key.astype(str).map(location_segment_map)
    fine_segment = result["hour"].map(wm._classify_by_hour)
    coarse_segment = result["hour"].map(wm._classify_low_medium_by_hour)

    result["time_segment"] = fine_segment
    low_medium_mask = restaurant_segment.isin(wm.LOW_MEDIUM_SEGMENTS)
    result.loc[low_medium_mask, "time_segment"] = coarse_segment.loc[low_medium_mask]
    result.loc[~result["is_open"], "time_segment"] = "closed"
    return result


def build_feature_frame_until_forecast(forecast_start, forecast_finish):
    log_step("Creating DB engine")
    engine = wm.build_engine()
    log_step(f"Loading orders with limit={FORECAST_ORDERS_LIMIT:,}")
    orders_df = wm.load_orders(engine, limit=FORECAST_ORDERS_LIMIT)
    log_step("Loaded orders", orders_df)
    log_step("Loading work hours")
    work_hours_df = wm.load_work_hours_df(engine)
    log_step("Loaded work hours", work_hours_df if work_hours_df is not None else None)

    log_step("Preparing orders")
    orders_df = wm.prepare_data(orders_df)
    orders_df = orders_df[orders_df[wm.TIMESTAMP_COLUMN] < forecast_start].copy()
    log_step("Prepared and filtered orders", orders_df)

    log_step("Building location groups and segments")
    location_to_group, _ = wm.build_merged_location_groups()
    grouped_orders_df = wm.apply_location_grouping(orders_df, location_to_group)
    location_segments = wm.build_location_segments(grouped_orders_df)
    log_step("Built location segments", location_segments)
    location_segment_map = dict(
        zip(location_segments[wm.LOCATION_COLUMN], location_segments["segment"])
    )

    log_step("Building hourly dataset")
    hourly_df = wm.build_hourly_dataset(orders_df)
    log_step("Built hourly dataset", hourly_df)
    log_step("Expanding hourly grid")
    hourly_df = wm.expand_hourly_grid(hourly_df)
    log_step("Expanded hourly grid", hourly_df)
    log_step("Adding time features")
    hourly_df = wm.add_time_features(hourly_df)
    log_step("Added time features", hourly_df)
    log_step("Adding is_open flag")
    hourly_df = add_is_open_flag_fast(hourly_df, work_hours_df)
    log_step("Added is_open flag", hourly_df)
    log_step("Classifying time segments")
    hourly_df = add_time_segment_fast(
        hourly_df,
        location_segment_map,
        location_to_group,
    )
    log_step("Classified time segments", hourly_df)
    log_step("Applying location grouping")
    hourly_df = wm.apply_location_grouping(hourly_df, location_to_group)
    log_step("Aggregating grouped hourly rows")
    hourly_df = wm.aggregate_hourly_after_grouping(hourly_df)
    log_step("Aggregated hourly rows", hourly_df)
    hourly_df = hourly_df.merge(
        location_segments,
        on=wm.LOCATION_COLUMN,
        how="left",
    )
    log_step("Merged location segments into hourly rows", hourly_df)
    hourly_df = wm.enforce_time_segment_policy(hourly_df)
    log_step("Building historical segment dataset")
    historical_segment_df = wm.build_segment_dataset(hourly_df)
    log_step("Built historical segment dataset", historical_segment_df)

    segment_grid = expand_segment_grid_until(
        historical_segment_df,
        forecast_finish,
    )
    segment_grid = add_future_open_hours(
        segment_grid,
        historical_segment_df,
        forecast_start,
    )
    feature_df = wm.create_segment_features(segment_grid)
    log_step("Created segment features", feature_df)
    historical_feature_df = feature_df[
        feature_df["segment_datetime"] < forecast_start
    ].copy()
    log_step("Adding location stats from historical rows", historical_feature_df)
    feature_df = wm.add_location_stats(historical_feature_df, feature_df)
    log_step("Added location stats", feature_df)
    return feature_df


def main():
    forecast_start = parse_moscow_datetime(FORECAST_START)
    forecast_finish = parse_moscow_datetime(FORECAST_FINISH)
    log_step(f"Forecast range: {forecast_start} -> {forecast_finish} (finish exclusive)")
    if forecast_finish <= forecast_start:
        raise ValueError("FORECAST_FINISH must be greater than FORECAST_START")

    artifact = load_model_artifact(MODEL_ARTIFACT_PATH)
    feature_df = build_feature_frame_until_forecast(
        forecast_start,
        forecast_finish,
    )
    target_df = feature_df[
        (feature_df["segment_datetime"] >= forecast_start)
        & (feature_df["segment_datetime"] < forecast_finish)
    ].copy()
    log_step("Selected future target rows", target_df)
    if target_df.empty:
        raise RuntimeError("No future feature rows were generated.")

    cfg = artifact["config"]
    log_step("Running CatBoost predictions")
    forecast_df = wm._predict_with_model_bundles(
        artifact["model_bundles"],
        target_df,
        artifact["features"],
        artifact["cat_features"],
        use_log=cfg["use_log"],
        round_predictions=cfg.get("round_predictions", wm.ROUND_PREDICTIONS_TO_INT),
        apply_zero_rule=cfg.get("apply_zero_rule", True),
    )
    forecast_df["orders_count"] = np.nan
    forecast_df["error"] = np.nan
    forecast_df["abs_error"] = np.nan
    log_step("Generated future predictions", forecast_df)

    output_columns = [
        wm.LOCATION_COLUMN,
        "segment_datetime",
        "segment",
        "time_segment",
        "orders_count",
        "prediction",
        "error",
        "abs_error",
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast_df[output_columns].to_csv(OUTPUT_PATH, index=False)
    print(f"Saved future order forecast to: {OUTPUT_PATH}")
    print(
        forecast_df.groupby(["segment", "time_segment"])["prediction"]
        .agg(["size", "sum", "mean"])
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
