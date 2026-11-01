"""Convert forecast orders to courier slots with recently observed productivity.

For each (planning location, vehicle, day type, hour) we measure how many
orders one courier on shift handled during the last weeks:

    productivity = orders delivered / couriers on shift

Thin cells are shrunk towards the same location/vehicle/time_segment average.
Needed slots are then ``forecast orders / productivity``. The order forecast
stays the driver (more orders -> more couriers); the fact only provides the
conversion rate. Cells without usable recent data keep the model slots.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

try:
    from . import main as cm
    from . import staffing_corridor as sc
except ImportError:
    import main as cm
    import staffing_corridor as sc


RECENT_PRODUCTIVITY_ENABLED = os.getenv("RECENT_PRODUCTIVITY_SLOTS", "1") == "1"
PRODUCTIVITY_LOOKBACK_DAYS = int(os.getenv("PRODUCTIVITY_LOOKBACK_DAYS", "28"))
# Shrinkage weight, in courier-hours, towards the time_segment average.
PRODUCTIVITY_SHRINK_COURIER_HOURS = float(
    os.getenv("PRODUCTIVITY_SHRINK_COURIER_HOURS", "8")
)
PRODUCTIVITY_MIN = float(os.getenv("RECENT_PRODUCTIVITY_MIN", "0.5"))
PRODUCTIVITY_MAX = float(os.getenv("RECENT_PRODUCTIVITY_MAX", "12"))
# slots = ceil(orders / productivity - tolerance); 0 is a plain ceil.
SLOT_CEIL_TOLERANCE = float(os.getenv("SLOT_CEIL_TOLERANCE", "0.7"))
# Cells with fewer observed courier-hours fall back to the model slots.
MIN_COURIER_HOURS = float(os.getenv("PRODUCTIVITY_MIN_COURIER_HOURS", "4"))


def load_recent_orders_by_hour(forecast_start, location_to_planning):
    start, finish = sc.lookback_window(forecast_start, PRODUCTIVITY_LOOKBACK_DAYS)
    engine = cm.build_engine()
    orders = cm.load_order_vehicle_history(
        engine,
        cm.datetime_to_ms(start),
        cm.datetime_to_ms(finish),
    )
    columns = ["location_id", "date", "hour", "vehicle_type", "orders"]
    if orders.empty:
        return pd.DataFrame(columns=columns)
    orders = cm.apply_location_grouping(orders, location_to_planning)
    moscow = cm.to_moscow_datetime(orders[cm.TIMESTAMP_COLUMN])
    frame = pd.DataFrame(
        {
            "location_id": orders[cm.LOCATION_COLUMN].astype(str),
            "date": moscow.dt.tz_localize(None).dt.normalize(),
            "hour": moscow.dt.hour,
            "vehicle_type": orders["vehicle"].map(cm.vehicle_to_type),
        }
    ).dropna()
    return (
        frame.groupby(["location_id", "date", "hour", "vehicle_type"])
        .size()
        .reset_index(name="orders")
    )


def build_productivity_table(orders_hourly, couriers_hourly, hour_segments, lookback_dates):
    """Productivity per (location, vehicle, day type, time_segment).

    Day-type cells are shrunk towards the all-days value of the same
    location/vehicle/time_segment. Segment level (not hour level) keeps the hourly
    demand smooth, like the model's own slot demand.
    """
    dates = pd.DatetimeIndex(lookback_dates)
    date_day_type = pd.Series(sc.day_type_of(dates), index=dates)

    def tag(frame):
        frame = frame.copy()
        frame["location_id"] = frame["location_id"].astype(str)
        frame["day_type"] = frame["date"].map(date_day_type)
        return frame.merge(hour_segments, on=["location_id", "hour"], how="inner")

    orders = tag(orders_hourly)
    couriers = tag(couriers_hourly)
    keys = ["location_id", "vehicle_type", "time_segment"]
    by_day = (
        orders.groupby(keys + ["day_type"])["orders"].sum().rename("orders").to_frame()
        .join(
            couriers.groupby(keys + ["day_type"])["couriers"].sum().rename("courier_hours"),
            how="outer",
        )
        .fillna(0.0)
        .reset_index()
    )
    overall = by_day.groupby(keys, as_index=False)[["orders", "courier_hours"]].sum()
    overall["overall_productivity"] = np.where(
        overall["courier_hours"] > 0,
        overall["orders"] / overall["courier_hours"].replace(0, np.nan),
        np.nan,
    )
    table = by_day.merge(
        overall[keys + ["overall_productivity", "courier_hours"]].rename(
            columns={"courier_hours": "overall_courier_hours"}
        ),
        on=keys,
        how="left",
    )
    k = PRODUCTIVITY_SHRINK_COURIER_HOURS
    prior = table["overall_productivity"]
    shrunk = (table["orders"] + k * prior) / (table["courier_hours"] + k)
    own = table["orders"] / table["courier_hours"].replace(0, np.nan)
    table["productivity"] = shrunk.where(prior.notna(), own).clip(
        PRODUCTIVITY_MIN, PRODUCTIVITY_MAX
    )
    table.loc[table["overall_courier_hours"].fillna(0) < MIN_COURIER_HOURS, "productivity"] = np.nan
    return table[keys + ["day_type", "productivity", "orders", "courier_hours"]]


def apply_recent_productivity(hourly_df, table):
    """Replace ``slots_needed`` by mean hourly orders / recent productivity per segment."""
    if hourly_df.empty or table is None or table.empty:
        return hourly_df, pd.DataFrame()
    hourly = hourly_df.copy()
    hourly["location_id"] = hourly["location_id"].astype(str)
    hourly["_day_type"] = sc.day_type_of(hourly["date"])
    keyed = table.rename(columns={"day_type": "_day_type"})
    hourly = hourly.merge(
        keyed[["location_id", "vehicle_type", "time_segment", "_day_type", "productivity"]],
        on=["location_id", "vehicle_type", "time_segment", "_day_type"],
        how="left",
    )
    positive = hourly["predicted_orders"].where(hourly["predicted_orders"] > 0)
    mean_orders = positive.groupby(
        [hourly["location_id"], hourly["date"], hourly["vehicle_type"], hourly["time_segment"]]
    ).transform("mean")
    has = hourly["productivity"].notna()
    need = mean_orders.fillna(0.0) / hourly["productivity"]
    new_slots = np.ceil(need - SLOT_CEIL_TOLERANCE).clip(lower=0)

    original = hourly["slots_needed"].astype(int)
    hourly["slots_needed_model"] = original
    hourly["slots_needed"] = np.where(has, new_slots, original).astype(int)
    changed = hourly["slots_needed"] != hourly["slots_needed_model"]
    if "slot_demand_source" in hourly.columns:
        hourly.loc[changed, "slot_demand_source"] = (
            hourly.loc[changed, "slot_demand_source"].astype(str) + "+recent_productivity"
        )
    audit = (
        hourly.assign(
            raised=np.where(hourly["slots_needed"] > hourly["slots_needed_model"],
                            hourly["slots_needed"] - hourly["slots_needed_model"], 0),
            lowered=np.where(hourly["slots_needed"] < hourly["slots_needed_model"],
                             hourly["slots_needed_model"] - hourly["slots_needed"], 0),
            covered=has.astype(int),
        )
        .groupby(["location_id", "vehicle_type"], as_index=False)
        .agg(
            hour_rows=("covered", "size"),
            rows_with_productivity=("covered", "sum"),
            slot_hours_model=("slots_needed_model", "sum"),
            slot_hours_after=("slots_needed", "sum"),
            slot_hours_raised=("raised", "sum"),
            slot_hours_lowered=("lowered", "sum"),
        )
    )
    return hourly.drop(columns=["_day_type", "productivity"]), audit
