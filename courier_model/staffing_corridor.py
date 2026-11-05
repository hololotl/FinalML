"""Keep modelled courier demand inside a corridor of recently observed staffing.

For every (planning location, day type, hour, vehicle) we look at how many
couriers were really scheduled on the same day type during the last weeks.
The modelled ``slots_needed`` is then clamped between a low and a high
quantile of that history.

Only history *before* the forecast week is used.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from . import main as cm
except ImportError:
    import main as cm


STAFFING_CORRIDOR_ENABLED = os.getenv("STAFFING_CORRIDOR", "1") == "1"
CORRIDOR_LOOKBACK_DAYS = int(os.getenv("CORRIDOR_LOOKBACK_DAYS", "28"))
CORRIDOR_LOW_QUANTILE = float(os.getenv("CORRIDOR_LOW_QUANTILE", "0.25"))
CORRIDOR_HIGH_QUANTILE = float(os.getenv("CORRIDOR_HIGH_QUANTILE", "0.6"))
# A (location, day type, hour, vehicle) cell gets a corridor only if couriers
# were present on at least this share of the dates; rarer cells stay model-only.
CORRIDOR_MIN_ACTIVE_SHARE = float(os.getenv("CORRIDOR_MIN_ACTIVE_SHARE", "0.25"))
# Soften the corridor: lower bound - margin (never below 0), upper bound + margin.
CORRIDOR_SOFT_MARGIN = int(os.getenv("CORRIDOR_SOFT_MARGIN", "0"))
CORRIDOR_MIN_HOUR_OVERLAP = float(os.getenv("CORRIDOR_MIN_HOUR_OVERLAP", "0.5"))

# Dynamic corridor: shift the window when the forecast orders of a restaurant
# differ from its recent orders more than the network as a whole does.
CORRIDOR_DYNAMIC = os.getenv("CORRIDOR_DYNAMIC", "0") == "1"
CORRIDOR_DYNAMIC_NORMALIZE = os.getenv("CORRIDOR_DYNAMIC_NORMALIZE", "1") == "1"
CORRIDOR_DYNAMIC_TOLERANCE = float(os.getenv("CORRIDOR_DYNAMIC_TOLERANCE", "0.1"))
CORRIDOR_DYNAMIC_MIN_DAILY_ORDERS = float(
    os.getenv("CORRIDOR_DYNAMIC_MIN_DAILY_ORDERS", "3")
)
CORRIDOR_DYNAMIC_MAX_SCALE = float(os.getenv("CORRIDOR_DYNAMIC_MAX_SCALE", "2.0"))

# Drop planned slots in hours where no courier worked during recent history.
RECENT_HOUR_STAFFING_GATE = os.getenv("RECENT_HOUR_STAFFING_GATE", "1") == "1"
RECENT_HOUR_STAFFING_LOOKBACK_DAYS = int(
    os.getenv("RECENT_HOUR_STAFFING_LOOKBACK_DAYS", "7")
)

# TODO(REMOVE): Ops feedback — ``work_hours`` in DB runs later than real closing.
# Fix prod ``work_hours`` and delete this block + ``apply_feedback_closing_hour_clamp``.
FEEDBACK_CLOSING_HOUR_CLAMP = os.getenv("FEEDBACK_CLOSING_HOUR_CLAMP", "1") == "1"
# ``hour`` is slot start; first hour start to zero (inclusive) when restaurant is closed.
FEEDBACK_LOCATION_FIRST_ZERO_HOUR = {
    "115": 22,  # Опен Символ — closes 22:00
    "126": 22,  # Опен Покровка — closes 22:00
    "39": 21,  # Гребной — closes 21:00
}

CORRIDOR_COLUMNS = [
    "location_id",
    "vehicle_type",
    "hour",
    "day_type",
    "corridor_lo",
    "corridor_hi",
    "active_share",
    "history_dates",
]


def day_type_of(dates):
    dates = pd.to_datetime(pd.Series(dates))
    return np.where(dates.dt.dayofweek >= 5, "weekend", "weekday")


def lookback_window(forecast_start, lookback_days=None):
    lookback_days = CORRIDOR_LOOKBACK_DAYS if lookback_days is None else lookback_days
    finish = pd.Timestamp(forecast_start)
    if finish.tzinfo is None:
        finish = finish.tz_localize("Europe/Moscow")
    finish = finish.tz_convert("Europe/Moscow").normalize()
    start = finish - pd.Timedelta(days=lookback_days)
    return start, finish


def schedule_hour_counts(schedule_df, start, finish, min_overlap=None):
    """Couriers on shift per (location, date, hour, vehicle) inside [start, finish)."""
    min_overlap = CORRIDOR_MIN_HOUR_OVERLAP if min_overlap is None else min_overlap
    columns = ["location_id", "date", "hour", "vehicle_type", "couriers"]
    if schedule_df is None or schedule_df.empty:
        return pd.DataFrame(columns=columns)

    span_hours = (finish - start).total_seconds() / 3600.0
    starts = (
        (schedule_df["schedule_start_dt"] - start).dt.total_seconds() / 3600.0
    ).clip(lower=0).to_numpy()
    finishes = (
        (schedule_df["schedule_finish_dt"] - start).dt.total_seconds() / 3600.0
    ).clip(upper=span_hours).to_numpy()
    valid = finishes > starts
    if not valid.any():
        return pd.DataFrame(columns=columns)

    locations = schedule_df[cm.LOCATION_COLUMN].astype(str).to_numpy()[valid]
    vehicles = schedule_df["vehicle_type"].astype(str).to_numpy()[valid]
    starts = starts[valid]
    finishes = finishes[valid]

    first_hour = np.floor(starts).astype(int)
    counts = (np.ceil(finishes).astype(int) - first_hour).clip(min=0)
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame(columns=columns)
    row_index = np.repeat(np.arange(len(starts)), counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    absolute_hour = first_hour[row_index] + offsets
    overlap = np.minimum(finishes[row_index], absolute_hour + 1.0) - np.maximum(
        starts[row_index], absolute_hour.astype(float)
    )
    keep = overlap >= min_overlap - 1e-9
    if not keep.any():
        return pd.DataFrame(columns=columns)

    absolute_hour = absolute_hour[keep]
    row_index = row_index[keep]
    base_date = pd.Timestamp(start.date())
    result = pd.DataFrame(
        {
            "location_id": locations[row_index],
            "date": base_date + pd.to_timedelta(absolute_hour // 24, unit="D"),
            "hour": (absolute_hour % 24).astype(int),
            "vehicle_type": vehicles[row_index],
        }
    )
    return (
        result.groupby(["location_id", "date", "hour", "vehicle_type"])
        .size()
        .reset_index(name="couriers")
    )


def load_recent_staffing_counts(forecast_start, location_to_planning, lookback_days=None):
    start, finish = lookback_window(forecast_start, lookback_days)
    engine = cm.build_engine()
    schedule_df = cm.load_schedule_history(
        engine,
        cm.datetime_to_ms(start),
        cm.datetime_to_ms(finish),
    )
    schedule_df = cm.prepare_schedule_history(schedule_df, location_to_planning)
    counts = schedule_hour_counts(schedule_df, start, finish)
    dates = pd.date_range(
        pd.Timestamp(start.date()),
        pd.Timestamp(finish.date()),
        inclusive="left",
    )
    return counts, dates


def build_staffing_corridor(
    counts,
    lookback_dates,
    low_quantile=None,
    high_quantile=None,
    min_active_share=None,
    soft_margin=None,
):
    soft_margin = CORRIDOR_SOFT_MARGIN if soft_margin is None else soft_margin
    low_quantile = CORRIDOR_LOW_QUANTILE if low_quantile is None else low_quantile
    high_quantile = CORRIDOR_HIGH_QUANTILE if high_quantile is None else high_quantile
    min_active_share = (
        CORRIDOR_MIN_ACTIVE_SHARE if min_active_share is None else min_active_share
    )
    if counts is None or counts.empty:
        return pd.DataFrame(columns=CORRIDOR_COLUMNS)

    dates = pd.DatetimeIndex(lookback_dates)
    wide = counts.pivot_table(
        index=["location_id", "vehicle_type", "hour"],
        columns="date",
        values="couriers",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=dates, fill_value=0)
    values = wide.to_numpy(dtype=float)
    index = wide.index.to_frame(index=False)

    frames = []
    for day_type in ["weekday", "weekend"]:
        columns = np.where(day_type_of(dates) == day_type)[0]
        if len(columns) == 0:
            continue
        sample = values[:, columns]
        active_share = (sample > 0).mean(axis=1)
        low = np.floor(np.quantile(sample, low_quantile, axis=1))
        high = np.ceil(np.quantile(sample, high_quantile, axis=1))
        high = np.maximum(high, 1.0)
        low = np.minimum(low, high)
        if soft_margin:
            low = np.maximum(low - soft_margin, 0.0)
            high = high + soft_margin
        frame = index.copy()
        frame["day_type"] = day_type
        frame["corridor_lo"] = low.astype(int)
        frame["corridor_hi"] = high.astype(int)
        frame["active_share"] = np.round(active_share, 3)
        frame["history_dates"] = len(columns)
        frames.append(frame[active_share >= min_active_share - 1e-9])
    if not frames:
        return pd.DataFrame(columns=CORRIDOR_COLUMNS)
    return pd.concat(frames, ignore_index=True)[CORRIDOR_COLUMNS]


def recent_daily_orders(orders_hourly, hour_segments, lookback_dates):
    """Mean daily orders per (location, day_type, time_segment) in the lookback."""
    columns = ["location_id", "day_type", "time_segment", "recent_daily_orders"]
    if orders_hourly is None or orders_hourly.empty:
        return pd.DataFrame(columns=columns)
    dates = pd.DatetimeIndex(lookback_dates)
    date_day_type = pd.Series(day_type_of(dates), index=dates)
    frame = orders_hourly.copy()
    frame["location_id"] = frame["location_id"].astype(str)
    frame = frame.merge(hour_segments, on=["location_id", "hour"], how="inner")
    frame["day_type"] = frame["date"].map(date_day_type)
    totals = (
        frame.groupby(["location_id", "day_type", "time_segment"])["orders"]
        .sum()
        .reset_index()
    )
    day_counts = date_day_type.value_counts().to_dict()
    totals["recent_daily_orders"] = totals["orders"] / totals["day_type"].map(
        day_counts
    ).clip(lower=1)
    return totals[columns]


def build_trend_ratio(hourly_df, recent):
    """forecast / recent daily orders per (location, day_type, time_segment).

    With normalisation the network-wide forecast/recent ratio is divided out, so
    a systematic over- or under-forecast does not move every corridor.
    """
    columns = ["location_id", "day_type", "time_segment", "trend_ratio"]
    if recent is None or recent.empty:
        return pd.DataFrame(columns=columns)
    frame = hourly_df[["location_id", "date", "time_segment", "predicted_orders"]].copy()
    frame["location_id"] = frame["location_id"].astype(str)
    frame["day_type"] = day_type_of(frame["date"])
    per_date = frame.groupby(
        ["location_id", "date", "day_type", "time_segment"], as_index=False
    )["predicted_orders"].sum()
    forecast = (
        per_date.groupby(["location_id", "day_type", "time_segment"], as_index=False)[
            "predicted_orders"
        ]
        .mean()
        .rename(columns={"predicted_orders": "forecast_daily_orders"})
    )
    trend = forecast.merge(
        recent, on=["location_id", "day_type", "time_segment"], how="inner"
    )
    trend = trend[trend["recent_daily_orders"] >= CORRIDOR_DYNAMIC_MIN_DAILY_ORDERS].copy()
    if trend.empty:
        return pd.DataFrame(columns=columns)
    trend["trend_ratio"] = trend["forecast_daily_orders"] / trend["recent_daily_orders"]
    if CORRIDOR_DYNAMIC_NORMALIZE:
        network = trend["forecast_daily_orders"].sum() / trend["recent_daily_orders"].sum()
        trend["trend_ratio"] = trend["trend_ratio"] / network
    return trend[columns]


def apply_staffing_corridor(hourly_df, corridor_df, trend_df=None):
    """Clamp ``slots_needed`` to the corridor; returns (hourly_df, audit_df)."""
    if hourly_df.empty or corridor_df is None or corridor_df.empty:
        return hourly_df, pd.DataFrame()

    hourly = hourly_df.copy()
    hourly["location_id"] = hourly["location_id"].astype(str)
    hourly["_day_type"] = day_type_of(hourly["date"])
    corridor = corridor_df.rename(columns={"day_type": "_day_type"})[
        [
            "location_id",
            "vehicle_type",
            "hour",
            "_day_type",
            "corridor_lo",
            "corridor_hi",
        ]
    ].copy()
    corridor["location_id"] = corridor["location_id"].astype(str)
    hourly["hour"] = hourly["hour"].astype(int)
    hourly = hourly.merge(
        corridor,
        on=["location_id", "vehicle_type", "hour", "_day_type"],
        how="left",
    )

    if CORRIDOR_DYNAMIC and trend_df is not None and not trend_df.empty:
        hourly = hourly.merge(
            trend_df.rename(columns={"day_type": "_day_type"}),
            on=["location_id", "_day_type", "time_segment"],
            how="left",
        )
        scale = hourly["trend_ratio"].fillna(1.0).clip(
            1.0 / CORRIDOR_DYNAMIC_MAX_SCALE, CORRIDOR_DYNAMIC_MAX_SCALE
        )
        moved = (scale - 1.0).abs() > CORRIDOR_DYNAMIC_TOLERANCE
        hourly["corridor_lo"] = np.where(
            moved, np.floor(hourly["corridor_lo"] * scale + 0.5), hourly["corridor_lo"]
        )
        hourly["corridor_hi"] = np.where(
            moved, np.ceil(hourly["corridor_hi"] * scale - 1e-9), hourly["corridor_hi"]
        )
        hourly["corridor_hi"] = np.maximum(hourly["corridor_hi"], hourly["corridor_lo"])
    has_corridor = hourly["corridor_hi"].notna()
    low = hourly["corridor_lo"].fillna(0.0).to_numpy()
    high = hourly["corridor_hi"].fillna(1e6).to_numpy()

    original = hourly["slots_needed"].astype(int)
    clamped = np.where(
        has_corridor,
        np.clip(original.to_numpy(), low, high),
        original.to_numpy(),
    ).astype(int)
    hourly["slots_needed_model"] = original
    hourly["slots_needed"] = clamped
    changed = hourly["slots_needed"] != hourly["slots_needed_model"]
    if "slot_demand_source" in hourly.columns:
        hourly.loc[changed, "slot_demand_source"] = (
            hourly.loc[changed, "slot_demand_source"].astype(str) + "+corridor"
        )

    audit = (
        hourly.assign(
            raised=np.where(
                changed & (hourly["slots_needed"] > hourly["slots_needed_model"]),
                hourly["slots_needed"] - hourly["slots_needed_model"],
                0,
            ),
            lowered=np.where(
                changed & (hourly["slots_needed"] < hourly["slots_needed_model"]),
                hourly["slots_needed_model"] - hourly["slots_needed"],
                0,
            ),
        )
        .groupby(["location_id", "vehicle_type"], as_index=False)
        .agg(
            slot_hours_model=("slots_needed_model", "sum"),
            slot_hours_after_corridor=("slots_needed", "sum"),
            slot_hours_raised=("raised", "sum"),
            slot_hours_lowered=("lowered", "sum"),
        )
    )
    hourly = hourly.drop(columns=["_day_type"])
    return hourly, audit


def build_recent_active_hours(counts):
    """(location_id, hour) pairs with at least one staffed hour in ``counts``."""
    if counts is None or counts.empty:
        return set(), set()
    active = counts[counts["couriers"] > 0].drop_duplicates(
        ["location_id", "date", "hour"]
    )
    keys = {
        (str(row.location_id), int(row.hour))
        for row in active.itertuples(index=False)
    }
    locations_with_history = set(counts["location_id"].astype(str).unique())
    return keys, locations_with_history


def apply_recent_hour_staffing_gate(hourly_df, active_hours, locations_with_history):
    """Zero ``slots_needed`` when the hour never had couriers in recent history."""
    if hourly_df.empty or not locations_with_history:
        return hourly_df, pd.DataFrame()

    hourly = hourly_df.copy()
    hourly["location_id"] = hourly["location_id"].astype(str)
    hourly["hour"] = hourly["hour"].astype(int)
    if "slots_needed_model" not in hourly.columns:
        hourly["slots_needed_model"] = hourly["slots_needed"]

    def allowed(row):
        location_id = row["location_id"]
        if location_id not in locations_with_history:
            return True
        return (location_id, int(row["hour"])) in active_hours

    keep = hourly.apply(allowed, axis=1)
    blocked = ~keep & (hourly["slots_needed"].astype(int) > 0)
    before = hourly["slots_needed"].astype(int)
    hourly.loc[blocked, "slots_needed"] = 0
    if "slot_demand_source" in hourly.columns:
        hourly.loc[blocked, "slot_demand_source"] = (
            hourly.loc[blocked, "slot_demand_source"].astype(str)
            + "+no_recent_hour"
        )

    audit_rows = []
    for (location_id, hour), group in hourly.groupby(["location_id", "hour"]):
        if location_id not in locations_with_history:
            continue
        slot_hours_before = int(before.loc[group.index].sum())
        slot_hours_after = int(group["slots_needed"].astype(int).sum())
        if slot_hours_before == slot_hours_after:
            continue
        audit_rows.append(
            {
                "location_id": location_id,
                "hour": int(hour),
                "had_recent_staffing": (location_id, int(hour)) in active_hours,
                "slot_hours_before": slot_hours_before,
                "slot_hours_after": slot_hours_after,
            }
        )
    audit = pd.DataFrame(audit_rows)
    return hourly, audit


def apply_feedback_closing_hour_clamp(hourly_df):
    """Zero slots at/after ops-reported closing for a fixed location list."""
    if hourly_df.empty or not FEEDBACK_CLOSING_HOUR_CLAMP:
        return hourly_df, pd.DataFrame()

    hourly = hourly_df.copy()
    hourly["location_id"] = hourly["location_id"].astype(str)
    hourly["hour"] = hourly["hour"].astype(int)
    if "slots_needed_model" not in hourly.columns:
        hourly["slots_needed_model"] = hourly["slots_needed"]

    audit_rows = []
    for location_id, first_zero in FEEDBACK_LOCATION_FIRST_ZERO_HOUR.items():
        mask = (hourly["location_id"] == location_id) & (
            hourly["hour"] >= int(first_zero)
        )
        if not mask.any():
            continue
        before = hourly.loc[mask, "slots_needed"].astype(int).sum()
        if before <= 0:
            continue
        hourly.loc[mask, "slots_needed"] = 0
        if "slot_demand_source" in hourly.columns:
            hourly.loc[mask, "slot_demand_source"] = (
                hourly.loc[mask, "slot_demand_source"].astype(str)
                + "+feedback_close"
            )
        audit_rows.append(
            {
                "location_id": location_id,
                "first_zero_hour": int(first_zero),
                "slot_hours_removed": int(before),
            }
        )
    return hourly, pd.DataFrame(audit_rows)
