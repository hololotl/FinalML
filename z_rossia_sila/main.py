import os
import importlib.util

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from catboost import CatBoostRegressor

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://courier:1337@localhost:1338/coffee"
)
TABLE_NAME = os.getenv("ORDERS_TABLE", "orders")
TIMESTAMP_COLUMN = os.getenv("ORDERS_TIMESTAMP_COLUMN", "delivering_at")
LOCATION_COLUMN = os.getenv("ORDERS_RESTAURANT_COLUMN", "location_id")
MIN_ROWS_FOR_ACCEPTANCE = int(os.getenv("MIN_ROWS_FOR_ACCEPTANCE", "100"))
TARGET_MAX_MEAN_ABS_PCT_ERROR = float(
    os.getenv("TARGET_MAX_MEAN_ABS_PCT_ERROR", "30")
)
TRAIN_VALID_SPLIT_QUANTILE = float(os.getenv("TRAIN_VALID_SPLIT_QUANTILE", "0.9"))
ROUND_PREDICTIONS_TO_INT = os.getenv("ROUND_PREDICTIONS_TO_INT", "0") == "1"
print(ROUND_PREDICTIONS_TO_INT)
UNDERPREDICT_SEGMENT_TIME = [
    ("mega", "deep_night"),
    ("mega", "opening"),
    ("mega", "dinner"),
    ("medium", "dinner"),
]
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
RUSSIAN_FIXED_HOLIDAYS = {
    (1, 1),   # New Year holidays
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),   # Orthodox Christmas
    (1, 8),
    (2, 23),  # Defender of the Fatherland Day
    (3, 8),   # International Women's Day
    (5, 1),   # Spring and Labour Day
    (5, 9),   # Victory Day
    (6, 12),  # Russia Day
    (11, 4),  # National Unity Day
    (12, 31), # New Year's Eve
}

def build_engine():
    try:
        return create_engine(DB_URL)
    except ModuleNotFoundError as exc:
        if "psycopg2" not in str(exc) and "psycopg" not in str(exc):
            raise

        has_psycopg3 = importlib.util.find_spec("psycopg") is not None
        has_psycopg2 = importlib.util.find_spec("psycopg2") is not None

        install_hint = (
            "Install a PostgreSQL driver in this virtualenv:\n"
            "  .venv/bin/pip install psycopg2-binary\n"
            "or\n"
            "  .venv/bin/pip install psycopg[binary]"
        )
        if has_psycopg3 and DB_URL.startswith("postgresql://"):
            install_hint += (
                "\n\nDetected psycopg (v3). You can also use:\n"
                "  DB_URL=postgresql+psycopg://user:pass@host:port/dbname"
            )
        if has_psycopg2:
            install_hint += "\n\nDetected psycopg2; check DB_URL and environment."

        raise RuntimeError(
            f"PostgreSQL driver is missing for SQLAlchemy URL '{DB_URL}'.\n"
            f"{install_hint}"
        ) from exc

def load_work_hours_df(engine):
    wh_sql = (
        "SELECT location_id, weekday, working, start_hour, start_minutes, "
        "finish_hour, finish_minutes FROM work_hours where working = true"
    )
    try:
        work_hours_df = pd.read_sql_query(text(wh_sql), engine)
    except Exception:
        return None
    if work_hours_df.empty:
        return None
    work_hours_df["location_id"] = work_hours_df["location_id"].astype(str)
    return work_hours_df

def load_orders(engine, limit=None):
    query = [
        f"SELECT {LOCATION_COLUMN}, {TIMESTAMP_COLUMN} FROM {TABLE_NAME}",
    ]
    query.append(f"ORDER BY {TIMESTAMP_COLUMN} DESC")
    if limit:
        query.append(f"LIMIT {int(limit)}")

    sql = " ".join(query)
    return pd.read_sql_query(text(sql), engine)

def load_locations(engine):
    sql = "SELECT id AS location_id, name FROM locations"
    locations_df = pd.read_sql_query(text(sql), engine)
    locations_df["location_id"] = locations_df["location_id"].astype(str)
    return locations_df

def prepare_data(df):
    df = df.copy()

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        unit="ms",
        errors="coerce",
        utc=True
    )

    df[TIMESTAMP_COLUMN] = (
        df[TIMESTAMP_COLUMN]
        .dt.tz_convert("Europe/Moscow")
    )

    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT

    df = df.dropna(
        subset=[
            LOCATION_COLUMN,
            TIMESTAMP_COLUMN
        ]
    )

    df = df[
        (df[TIMESTAMP_COLUMN].dt.year >= 2024)
        & (df[TIMESTAMP_COLUMN].dt.year <= 2027)
    ]

    df[LOCATION_COLUMN] = df[LOCATION_COLUMN].astype(str)

    df["datetime_hour"] = (
        df[TIMESTAMP_COLUMN]
        .dt.floor("h")
    )

    return df


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
        union(str(left_id), str(right_id))

    members_by_root = {}
    for left_id, right_id, _ in MERGED_LOCATION_PAIRS:
        for location_id in [str(left_id), str(right_id)]:
            root = find(location_id)
            members_by_root.setdefault(root, set()).add(location_id)

    location_to_group = {}
    group_to_members = {}
    for members in members_by_root.values():
        sorted_members = sorted(members, key=int)
        group_id = "grp_" + "_".join(sorted_members)
        group_to_members[group_id] = sorted_members
        for member in sorted_members:
            location_to_group[member] = group_id

    return location_to_group, group_to_members


def apply_location_grouping(df, location_to_group):
    df = df.copy()
    df[LOCATION_COLUMN] = (
        df[LOCATION_COLUMN]
        .astype(str)
        .map(lambda x: location_to_group.get(x, x))
    )
    return df


def aggregate_hourly_after_grouping(hourly_df):
    grouped = (
        hourly_df.groupby(
            [
                LOCATION_COLUMN,
                "datetime_hour",
                "hour",
                "weekday",
                "month",
                "is_weekend",
                "is_holiday",
                "time_segment",
            ],
            as_index=False
        )
        .agg(
            orders_count=("orders_count", "sum"),
            is_open=("is_open", "sum"),
        )
    )
    return grouped

def add_time_features(df):
    df["hour"] = df["datetime_hour"].dt.hour
    df["weekday"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["is_weekend"] = (
        df["weekday"]
        .isin([5, 6])
        .astype(int)
    )
    df["is_holiday"] = (
        list(zip(df["month"], df["datetime_hour"].dt.day))
    )
    df["is_holiday"] = (
        pd.Series(df["is_holiday"], index=df.index)
        .isin(RUSSIAN_FIXED_HOLIDAYS)
        .astype(int)
    )
    return df

def build_location_segments(df):
    location_stats = (
        df.groupby(LOCATION_COLUMN)
        .size()
        .reset_index(name="orders_count")
    )

    p50 = location_stats["orders_count"].quantile(0.50)
    p90 = location_stats["orders_count"].quantile(0.85)
    p99 = location_stats["orders_count"].quantile(0.95)

    location_stats["segment"] = (
        location_stats["orders_count"]
        .apply(
            lambda x: classify_location(
                x,
                p50,
                p90,
                p99
            )
        )
    )

    return location_stats[
        [LOCATION_COLUMN, "segment"]
    ]

def classify_location(order_count, p50, p90, p99):
    if order_count <= p50:
        return "low"

    if order_count <= p90:
        return "medium"

    if order_count <= p99:
        return "high"

    return "mega"

def build_hourly_dataset(df):
    hourly_df = (
        df.groupby(
            [LOCATION_COLUMN, "datetime_hour"]
        )
        .size()
        .reset_index(name="orders_count")
    )

    return hourly_df

def build_hourly_dataset(df):
    hourly_df = (
        df.groupby(
            [LOCATION_COLUMN, "datetime_hour"]
        )
        .size()
        .reset_index(name="orders_count")
    )

    return hourly_df

def expand_hourly_grid(hourly_df):
    hourly_df = (
        hourly_df
        .set_index([LOCATION_COLUMN, "datetime_hour"])
        .sort_index()
    )

    def _expand(group):
        dt_index = group.index.get_level_values(1)
        idx = pd.date_range(
            dt_index.min(),
            dt_index.max(),
            freq="h",
            tz=dt_index.tz
        )
        loc = group.index.get_level_values(0)[0]
        new_index = pd.MultiIndex.from_product(
            [[loc], idx],
            names=[LOCATION_COLUMN, "datetime_hour"]
        )
        return group.reindex(new_index)

    expanded = (
        hourly_df
        .groupby(level=0, group_keys=False)
        .apply(_expand)
        .reset_index()
    )

    expanded["orders_count"] = expanded["orders_count"].fillna(0)
    return expanded

def add_time_features(df):
    df["hour"] = df["datetime_hour"].dt.hour
    df["weekday"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["is_weekend"] = (
        df["weekday"]
        .isin([5, 6])
        .astype(int)
    )
    df["is_holiday"] = (
        list(zip(df["month"], df["datetime_hour"].dt.day))
    )
    df["is_holiday"] = (
        pd.Series(df["is_holiday"], index=df.index)
        .isin(RUSSIAN_FIXED_HOLIDAYS)
        .astype(int)
    )
    return df

def is_open_hour(location_id, weekday, hour, work_hours_df):
    if work_hours_df is None or work_hours_df.empty:
        return True

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id)
        & (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return False

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    if start_hour < finish_hour:
        return start_hour <= hour < finish_hour

    return hour >= start_hour or hour < finish_hour

def add_is_open_flag(df, work_hours_df):
    df["is_open"] = df.apply(
        lambda row: is_open_hour(
            row[LOCATION_COLUMN],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    return df

def add_time_segment(df, work_hours_df):
    df["time_segment"] = df.apply(
        lambda row: classify_hour(
            row[LOCATION_COLUMN],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    return df

def _classify_by_hour(hour):
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

def classify_hour(location_id, weekday, hour, work_hours_df):
    if work_hours_df is None or work_hours_df.empty:
        return _classify_by_hour(hour)

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id)
        & (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return "closed"

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    if start_hour < finish_hour:
        is_open = start_hour <= hour < finish_hour
    else:
        is_open = hour >= start_hour or hour < finish_hour

    if not is_open:
        return "closed"

    return _classify_by_hour(hour)

SEGMENT_ORDER = [
    "deep_night",
    "opening",
    "early_morning",
    "morning",
    "lunch",
    "afternoon",
    "dinner",
    "late_evening",
]
SEGMENT_RANK = {name: idx for idx, name in enumerate(SEGMENT_ORDER)}


def build_segment_dataset(hourly_df):
    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["datetime_hour"].dt.floor("D")
    hourly_df["segment_rank"] = hourly_df["time_segment"].map(SEGMENT_RANK)

    segment_df = (
        hourly_df.groupby(
            [LOCATION_COLUMN, "segment", "date", "time_segment", "segment_rank"]
        )
        .agg(
            orders_count=("orders_count", "sum"),
            open_hours=("is_open", "sum"),
        )
        .reset_index()
    )

    return segment_df

def expand_segment_grid(segment_df):
    segments = pd.DataFrame({
        "time_segment": SEGMENT_ORDER,
        "segment_rank": [SEGMENT_RANK[s] for s in SEGMENT_ORDER],
    })

    grids = []
    for location_id, group in segment_df.groupby(LOCATION_COLUMN):
        location_segment = group["segment"].iloc[0]
        date_index = pd.date_range(
            group["date"].min(),
            group["date"].max(),
            freq="D"
        )
        grid = pd.MultiIndex.from_product(
            [date_index, segments["time_segment"]],
            names=["date", "time_segment"]
        ).to_frame(index=False)
        grid["segment_rank"] = grid["time_segment"].map(SEGMENT_RANK)
        grid[LOCATION_COLUMN] = location_id
        grid["segment"] = location_segment
        grids.append(grid)

    grid_df = pd.concat(grids, ignore_index=True)

    merged = grid_df.merge(
        segment_df,
        on=[LOCATION_COLUMN, "segment", "date", "time_segment", "segment_rank"],
        how="left"
    )

    merged["orders_count"] = merged["orders_count"].fillna(0)
    merged["open_hours"] = merged["open_hours"].fillna(0)

    merged["segment_datetime"] = (
        pd.to_datetime(merged["date"])
        + pd.to_timedelta(merged["segment_rank"], unit="h")
    )

    merged["weekday"] = pd.to_datetime(merged["date"]).dt.dayofweek
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    merged["is_weekend"] = (
        merged["weekday"]
        .isin([5, 6])
        .astype(int)
    )
    merged["is_holiday"] = (
        list(zip(merged["month"], pd.to_datetime(merged["date"]).dt.day))
    )
    merged["is_holiday"] = (
        pd.Series(merged["is_holiday"], index=merged.index)
        .isin(RUSSIAN_FIXED_HOLIDAYS)
        .astype(int)
    )

    return merged

def create_segment_features(df):
    df = df.copy()

    df = df.sort_values(
        by=[
            LOCATION_COLUMN,
            "segment_datetime"
        ]
    )

    df["open_orders"] = df["orders_count"].where(df["open_hours"] > 0)

    df["lag_1seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(1)
    )

    df["lag_2seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(2)
    )

    df["lag_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(8)
    )

    df["lag_56seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(56)
    )

    df["prev_open_1seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(1)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["prev_open_2seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(2)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["prev_open_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(8)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["rolling_mean_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    df["rolling_mean_7d"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(56)
            .mean()
        )
    )

    df["rolling_std_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .std()
        )
    )

    df["rolling_open_count_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .sum()
        )
    )

    df["rolling_max_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .max()
        )
    )

    df["rolling_max_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(8)
            .max()
        )
    )

    df["rolling_min_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .min()
        )
    )

    df["trend_3seg"] = df["rolling_mean_3seg"] - df["lag_8seg"]
    df["trend_ratio"] = df["lag_1seg"] / (df["lag_8seg"] + 1)
    df["surge_ratio"] = df["lag_1seg"] / (df["rolling_mean_3seg"] + 1)

    df["segment_sin"] = np.sin(2 * np.pi * df["segment_rank"] / 8)
    df["segment_cos"] = np.cos(2 * np.pi * df["segment_rank"] / 8)

    same_segment_group = df.groupby([LOCATION_COLUMN, "time_segment"])["open_orders"]
    df["lag_same_segment_1d"] = same_segment_group.shift(1)
    df["lag_same_segment_7d"] = same_segment_group.shift(7)
    df["rolling_same_segment_mean_3d"] = (
        same_segment_group
        .transform(lambda x: x.shift(1).rolling(3).mean())
    )
    df["rolling_same_segment_max_7d"] = (
        same_segment_group
        .transform(lambda x: x.shift(1).rolling(7).max())
    )

    for col in [
        "lag_1seg",
        "lag_2seg",
        "lag_8seg",
        "lag_56seg",
        "rolling_mean_3seg",
        "rolling_mean_7d",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
        "rolling_max_3seg",
        "rolling_max_8seg",
        "rolling_min_3seg",
        "trend_3seg",
        "trend_ratio",
        "surge_ratio",
        "lag_same_segment_1d",
        "lag_same_segment_7d",
        "rolling_same_segment_mean_3d",
        "rolling_same_segment_max_7d",
    ]:
        df[col] = df[col].fillna(0)

    df = df[df["open_hours"] > 0].copy()
    df = df.drop(columns=["open_orders"])

    return df

def split_train_test(df):
    split_date = df["segment_datetime"].quantile(0.8)

    train_df = (
        df[df["segment_datetime"] <= split_date]
    )

    test_df = (
        df[df["segment_datetime"] > split_date]
    )

    return train_df, test_df


def split_train_valid(train_df, quantile=TRAIN_VALID_SPLIT_QUANTILE):
    split_date = train_df["segment_datetime"].quantile(quantile)
    fit_df = train_df[train_df["segment_datetime"] <= split_date].copy()
    valid_df = train_df[train_df["segment_datetime"] > split_date].copy()
    return fit_df, valid_df


def add_location_stats(train_df, full_df):
    stats = (
        train_df.groupby(LOCATION_COLUMN)["orders_count"]
        .agg(
            location_mean_orders="mean",
            location_std_orders="std",
            location_max_orders="max",
        )
        .reset_index()
    )

    seg_stats = (
        train_df.groupby([LOCATION_COLUMN, "time_segment"])["orders_count"]
        .agg(
            location_seg_mean_orders="mean",
            location_seg_std_orders="std",
            location_seg_max_orders="max",
        )
        .reset_index()
    )

    seg_defaults = (
        train_df.groupby("time_segment")["orders_count"]
        .agg(
            seg_mean_orders="mean",
            seg_std_orders="std",
            seg_max_orders="max",
        )
        .to_dict()
    )

    full_df = full_df.merge(stats, on=LOCATION_COLUMN, how="left")
    full_df = full_df.merge(
        seg_stats,
        on=[LOCATION_COLUMN, "time_segment"],
        how="left"
    )

    full_df["location_mean_orders"] = full_df["location_mean_orders"].fillna(
        train_df["orders_count"].mean()
    )
    full_df["location_std_orders"] = full_df["location_std_orders"].fillna(
        train_df["orders_count"].std()
    )
    full_df["location_max_orders"] = full_df["location_max_orders"].fillna(
        train_df["orders_count"].max()
    )

    full_df["location_seg_mean_orders"] = full_df[
        "location_seg_mean_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_mean_orders"])
    )
    full_df["location_seg_std_orders"] = full_df[
        "location_seg_std_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_std_orders"])
    )
    full_df["location_seg_max_orders"] = full_df[
        "location_seg_max_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_max_orders"])
    )

    return full_df


CAT_FEATURES = ["time_segment", "segment", "weekday"]
def build_feature_matrix(df):
    raw_features = [
        "weekday",
        "month",
        "is_weekend",
        "is_holiday",
        "open_hours",
        "segment_rank",

        "lag_1seg",
        "lag_2seg",
        "lag_8seg",
        "lag_56seg",

        "prev_open_1seg",
        "prev_open_2seg",
        "prev_open_8seg",

        "rolling_mean_3seg",
        "rolling_mean_7d",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
        "rolling_max_3seg",
        "rolling_max_8seg",
        "rolling_min_3seg",

        "trend_3seg",
        "trend_ratio",
        "surge_ratio",
        "lag_same_segment_1d",
        "lag_same_segment_7d",
        "rolling_same_segment_mean_3d",
        "rolling_same_segment_max_7d",

        "segment_sin",
        "segment_cos",

        "location_mean_orders",
        "location_std_orders",
        "location_max_orders",

        "location_seg_mean_orders",
        "location_seg_std_orders",
        "location_seg_max_orders",
    ]

    cat_features = [f for f in CAT_FEATURES if f in df.columns]

    # Ensure categorical features are not duplicated in the numeric list.
    features = [f for f in raw_features if f not in cat_features]

    return features, cat_features


def compute_sample_weight(
    train_df,
    peak_quantile=0.97,
    peak_weight=1.8,
    underpredict_weight=1.0,
    underpredict_pairs=None
):
    if underpredict_pairs is None:
        underpredict_pairs = []

    peak_thresholds = (
        train_df.groupby("segment")["orders_count"]
        .quantile(peak_quantile)
        .to_dict()
    )

    peak_mask = train_df["orders_count"] >= train_df["segment"].map(peak_thresholds)
    sample_weight = np.ones(len(train_df), dtype=float)
    sample_weight = np.where(peak_mask, sample_weight * peak_weight, sample_weight)

    if underpredict_pairs:
        underpredict_pairs_set = set(underpredict_pairs)
        underpredict_mask = train_df.apply(
            lambda row: (row["segment"], row["time_segment"]) in underpredict_pairs_set,
            axis=1
        )
        sample_weight = np.where(
            underpredict_mask,
            sample_weight * underpredict_weight,
            sample_weight
        )

    return sample_weight


def train_catboost(
    train_df,
    valid_df,
    features,
    cat_features,
    use_log,
    experiment_cfg
):
    X_train = train_df[features + cat_features]
    y_train = train_df["orders_count"]
    X_valid = valid_df[features + cat_features]
    y_valid = valid_df["orders_count"]

    sample_weight = compute_sample_weight(
        train_df,
        peak_quantile=experiment_cfg["peak_quantile"],
        peak_weight=experiment_cfg["peak_weight"],
        underpredict_weight=experiment_cfg["underpredict_weight"],
        underpredict_pairs=experiment_cfg["underpredict_pairs"]
    )

    if use_log:
        y_train = np.log1p(y_train)
        y_valid = np.log1p(y_valid)

    cat_indices = [X_train.columns.get_loc(c) for c in cat_features]

    model = CatBoostRegressor(
        iterations=experiment_cfg["iterations"],
        learning_rate=experiment_cfg["learning_rate"],
        depth=experiment_cfg["depth"],
        l2_leaf_reg=experiment_cfg["l2_leaf_reg"],
        subsample=experiment_cfg["subsample"],
        random_strength=experiment_cfg["random_strength"],
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        od_type="Iter",
        od_wait=experiment_cfg["od_wait"],
        use_best_model=True,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_indices,
        sample_weight=sample_weight
    )

    return model

def predict_model(
    model,
    df,
    features,
    cat_features,
    use_log,
    clip_negative,
    round_predictions=ROUND_PREDICTIONS_TO_INT
):
    X = df[features + cat_features]
    pred = model.predict(X)
    if use_log:
        pred = np.expm1(pred)
    if clip_negative:
        pred = np.clip(pred, 0, None)
    if round_predictions:
        pred = np.rint(pred)

    result_df = df.copy()
    result_df["prediction"] = pred
    result_df["error"] = result_df["prediction"] - result_df["orders_count"]
    result_df["abs_error"] = result_df["error"].abs()

    return result_df

def evaluate_predictions(result_df):
    mae = mean_absolute_error(
        result_df["orders_count"],
        result_df["prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            result_df["orders_count"],
            result_df["prediction"]
        )
    )

    global_metrics = {
        "mae": mae,
        "rmse": rmse,
        "wape": (
            result_df["abs_error"].sum() / result_df["orders_count"].sum()
            if result_df["orders_count"].sum() > 0
            else np.nan
        ),
        "mean_abs_pct_error": (
            (
                result_df.loc[result_df["orders_count"] > 0, "abs_error"]
                / result_df.loc[result_df["orders_count"] > 0, "orders_count"]
            ).mean() * 100
            if (result_df["orders_count"] > 0).any()
            else np.nan
        ),
        "rows": len(result_df),
    }

    segment_metrics_df = (
        result_df.groupby("segment")
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "wape": (
                    g["abs_error"].sum() / g["orders_count"].sum()
                    if g["orders_count"].sum() > 0
                    else np.nan
                ),
                "mean_abs_pct_error": (
                    ((g.loc[g["orders_count"] > 0, "abs_error"]
                      / g.loc[g["orders_count"] > 0, "orders_count"]).mean() * 100)
                    if (g["orders_count"] > 0).any()
                    else np.nan
                ),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
    )

    time_segment_mae_df = (
        result_df.groupby("time_segment")
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "wape": (
                    g["abs_error"].sum() / g["orders_count"].sum()
                    if g["orders_count"].sum() > 0
                    else np.nan
                ),
            })
        )
        .reset_index()
    )

    segment_time_metrics_df = (
        result_df.groupby(["segment", "time_segment"])
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "wape": (
                    g["abs_error"].sum() / g["orders_count"].sum()
                    if g["orders_count"].sum() > 0
                    else np.nan
                ),
                "mean_abs_pct_error": (
                    ((g.loc[g["orders_count"] > 0, "abs_error"]
                      / g.loc[g["orders_count"] > 0, "orders_count"]).mean() * 100)
                    if (g["orders_count"] > 0).any()
                    else np.nan
                ),
                "rows_nonzero_orders": (g["orders_count"] > 0).sum(),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
    )
    segment_time_metrics_df["acceptance_applicable"] = (
        segment_time_metrics_df["rows"] >= MIN_ROWS_FOR_ACCEPTANCE
    )
    segment_time_metrics_df["acceptance_passed"] = (
        segment_time_metrics_df["acceptance_applicable"]
        & (
            segment_time_metrics_df["mean_abs_pct_error"]
            <= TARGET_MAX_MEAN_ABS_PCT_ERROR
        )
    )

    return global_metrics, segment_metrics_df, time_segment_mae_df, segment_time_metrics_df


def compute_peak_metrics(train_df, result_df):
    thresholds = (
        train_df.groupby("segment")["orders_count"]
        .quantile(0.95)
        .to_dict()
    )

    rows = []
    for segment, threshold in thresholds.items():
        seg_df = result_df[result_df["segment"] == segment]
        peaks = seg_df[seg_df["orders_count"] >= threshold]
        if len(peaks) == 0:
            rows.append({
                "segment": segment,
                "threshold": threshold,
                "rows": 0,
                "mae": np.nan,
                "rmse": np.nan,
            })
            continue

        rows.append({
            "segment": segment,
            "threshold": threshold,
            "rows": len(peaks),
            "mae": mean_absolute_error(peaks["orders_count"], peaks["prediction"]),
            "rmse": np.sqrt(mean_squared_error(peaks["orders_count"], peaks["prediction"])),
        })

    peak_metrics_df = pd.DataFrame(rows)

    peak_examples_df = (
        result_df
        .copy()
        .assign(peak_threshold=result_df["segment"].map(thresholds))
    )
    peak_examples_df = peak_examples_df[
        peak_examples_df["orders_count"] >= peak_examples_df["peak_threshold"]
    ]
    peak_examples_df = (
        peak_examples_df
        .sort_values("abs_error", ascending=False)
        .head(1000)
        [[
            LOCATION_COLUMN,
            "segment_datetime",
            "segment",
            "time_segment",
            "orders_count",
            "prediction",
            "error",
            "abs_error",
            "peak_threshold",
        ]]
    )

    return peak_metrics_df, peak_examples_df


def evaluate_group_predictions(result_df):
    merged_df = result_df[
        result_df[LOCATION_COLUMN].astype(str).str.startswith("grp_")
    ].copy()
    if merged_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    group_metrics_df = (
        merged_df.groupby(LOCATION_COLUMN)
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "wape": (
                    g["abs_error"].sum() / g["orders_count"].sum()
                    if g["orders_count"].sum() > 0
                    else np.nan
                ),
                "mean_abs_pct_error": (
                    ((g.loc[g["orders_count"] > 0, "abs_error"]
                      / g.loc[g["orders_count"] > 0, "orders_count"]).mean() * 100)
                    if (g["orders_count"] > 0).any()
                    else np.nan
                ),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
        .rename(columns={LOCATION_COLUMN: "merged_group"})
    )

    group_time_metrics_df = (
        merged_df.groupby([LOCATION_COLUMN, "time_segment"])
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "wape": (
                    g["abs_error"].sum() / g["orders_count"].sum()
                    if g["orders_count"].sum() > 0
                    else np.nan
                ),
                "mean_abs_pct_error": (
                    ((g.loc[g["orders_count"] > 0, "abs_error"]
                      / g.loc[g["orders_count"] > 0, "orders_count"]).mean() * 100)
                    if (g["orders_count"] > 0).any()
                    else np.nan
                ),
                "rows_nonzero_orders": (g["orders_count"] > 0).sum(),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
        .rename(columns={LOCATION_COLUMN: "merged_group"})
    )

    group_time_metrics_df["acceptance_applicable"] = (
        group_time_metrics_df["rows"] >= MIN_ROWS_FOR_ACCEPTANCE
    )
    group_time_metrics_df["acceptance_passed"] = (
        group_time_metrics_df["acceptance_applicable"]
        & (
            group_time_metrics_df["mean_abs_pct_error"]
            <= TARGET_MAX_MEAN_ABS_PCT_ERROR
        )
    )

    return group_metrics_df, group_time_metrics_df


def build_merged_group_predictions_df(result_df, group_to_members):
    merged_df = result_df[
        result_df[LOCATION_COLUMN].astype(str).str.startswith("grp_")
    ].copy()
    if merged_df.empty:
        return pd.DataFrame()

    merged_df["group_members"] = merged_df[LOCATION_COLUMN].map(
        lambda x: ",".join(group_to_members.get(str(x), []))
    )

    return (
        merged_df[
            [
                LOCATION_COLUMN,
                "group_members",
                "segment_datetime",
                "segment",
                "time_segment",
                "orders_count",
                "prediction",
                "error",
                "abs_error",
            ]
        ]
        .sort_values([LOCATION_COLUMN, "segment_datetime"])
    )


OUTPUT_DIR_LOG = "z_rossia_sila/res"


def build_experiment_configs():
    return [
        {
            "name": "exp01_log_baseline",
            "use_log": True,
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "subsample": 0.9,
            "random_strength": 1.0,
            "peak_quantile": 0.97,
            "peak_weight": 1.8,
            "underpredict_weight": 1.0,
            "underpredict_pairs": [],
            "od_wait": 120,
        },
        {
            "name": "exp02_log_peak3_under13",
            "use_log": True,
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "subsample": 0.9,
            "random_strength": 1.0,
            "peak_quantile": 0.97,
            "peak_weight": 3.0,
            "underpredict_weight": 1.3,
            "underpredict_pairs": UNDERPREDICT_SEGMENT_TIME,
            "od_wait": 120,
        },
        {
            "name": "exp03_log_peak4_under15",
            "use_log": True,
            "iterations": 2000,
            "learning_rate": 0.025,
            "depth": 10,
            "l2_leaf_reg": 7.0,
            "subsample": 0.85,
            "random_strength": 1.2,
            "peak_quantile": 0.97,
            "peak_weight": 4.0,
            "underpredict_weight": 1.5,
            "underpredict_pairs": UNDERPREDICT_SEGMENT_TIME,
            "od_wait": 140,
        },
        {
            "name": "exp04_no_log_peak4_under15",
            "use_log": False,
            "iterations": 2000,
            "learning_rate": 0.025,
            "depth": 8,
            "l2_leaf_reg": 7.0,
            "subsample": 0.85,
            "random_strength": 1.2,
            "peak_quantile": 0.97,
            "peak_weight": 4.0,
            "underpredict_weight": 1.5,
            "underpredict_pairs": UNDERPREDICT_SEGMENT_TIME,
            "od_wait": 140,
        },
    ]


def evaluate_acceptance(segment_time_metrics_df):
    check_df = segment_time_metrics_df[
        segment_time_metrics_df["acceptance_applicable"]
    ].copy()
    if check_df.empty:
        return {
            "applicable_rows": 0,
            "violations_count": 0,
            "violations_mean_excess_pct": 0.0,
            "pass_rate": np.nan,
        }

    violations = check_df[
        check_df["mean_abs_pct_error"] > TARGET_MAX_MEAN_ABS_PCT_ERROR
    ].copy()
    if not violations.empty:
        violations["excess_pct"] = (
            violations["mean_abs_pct_error"] - TARGET_MAX_MEAN_ABS_PCT_ERROR
        )

    return {
        "applicable_rows": len(check_df),
        "violations_count": len(violations),
        "violations_mean_excess_pct": (
            violations["excess_pct"].mean() if not violations.empty else 0.0
        ),
        "pass_rate": (
            (check_df["mean_abs_pct_error"] <= TARGET_MAX_MEAN_ABS_PCT_ERROR).mean()
        ),
    }


def print_top_worst_segment_time(segment_time_metrics_df, label, top_n=5):
    check_df = segment_time_metrics_df[
        segment_time_metrics_df["acceptance_applicable"]
    ].copy()
    if check_df.empty:
        print(f"\n=== TOP WORST SEGMENT_TIME ({label}) ===")
        print("No rows with acceptance_applicable=true.")
        return

    print(f"\n=== TOP WORST SEGMENT_TIME ({label}) ===")
    print(
        check_df
        .sort_values(["mean_abs_pct_error", "mae"], ascending=False)
        .head(top_n)
        [[
            "segment",
            "time_segment",
            "rows",
            "mean_abs_pct_error",
            "mae",
            "wape",
            "acceptance_passed",
        ]]
        .to_string(index=False)
    )


def run_experiments(train_df, test_df, features, cat_features):
    fit_df, valid_df = split_train_valid(train_df)
    experiment_rows = []
    best_result = None

    for cfg in build_experiment_configs():
        model = train_catboost(
            fit_df,
            valid_df,
            features,
            cat_features,
            use_log=cfg["use_log"],
            experiment_cfg=cfg
        )

        result_test = predict_model(
            model,
            test_df,
            features,
            cat_features,
            use_log=cfg["use_log"],
            clip_negative=True
        )
        result_train = predict_model(
            model,
            train_df,
            features,
            cat_features,
            use_log=cfg["use_log"],
            clip_negative=True
        )

        global_test, segment_test, _, segment_time_test = evaluate_predictions(result_test)
        global_train, _, _, _ = evaluate_predictions(result_train)
        peak_metrics, peak_examples = compute_peak_metrics(train_df, result_test)

        acceptance = evaluate_acceptance(segment_time_test)
        underpredict_penalty = (
            result_test.loc[result_test["error"] < 0, "abs_error"].mean()
            if (result_test["error"] < 0).any()
            else 0.0
        )
        peak_mae_mean = peak_metrics["mae"].mean(skipna=True)

        experiment_row = {
            "experiment_name": cfg["name"],
            "use_log": cfg["use_log"],
            "best_iteration": model.get_best_iteration(),
            "test_mae": global_test["mae"],
            "test_rmse": global_test["rmse"],
            "test_wape": global_test["wape"],
            "test_mean_abs_pct_error": global_test["mean_abs_pct_error"],
            "train_mae": global_train["mae"],
            "train_rmse": global_train["rmse"],
            "peak_mae_mean": peak_mae_mean,
            "underpredict_penalty": underpredict_penalty,
            "acceptance_applicable_rows": acceptance["applicable_rows"],
            "acceptance_violations": acceptance["violations_count"],
            "acceptance_mean_excess_pct": acceptance["violations_mean_excess_pct"],
            "acceptance_pass_rate": acceptance["pass_rate"],
        }
        experiment_rows.append(experiment_row)

        print(f"\n=== EXPERIMENT: {cfg['name']} ===")
        print(pd.DataFrame([experiment_row]).to_string(index=False))
        print_top_worst_segment_time(segment_time_test, cfg["name"], top_n=5)

        candidate = {
            "config": cfg,
            "model": model,
            "result_test": result_test,
            "global_test": global_test,
            "segment_test": segment_test,
            "segment_time_test": segment_time_test,
            "peak_metrics": peak_metrics,
            "peak_examples": peak_examples,
            "global_train": global_train,
            "experiment_row": experiment_row,
        }

        if best_result is None:
            best_result = candidate
            continue

        best_row = best_result["experiment_row"]
        challenger_key = (
            experiment_row["acceptance_violations"],
            experiment_row["acceptance_mean_excess_pct"],
            experiment_row["peak_mae_mean"],
            experiment_row["underpredict_penalty"],
            experiment_row["test_mae"],
        )
        best_key = (
            best_row["acceptance_violations"],
            best_row["acceptance_mean_excess_pct"],
            best_row["peak_mae_mean"],
            best_row["underpredict_penalty"],
            best_row["test_mae"],
        )
        if challenger_key < best_key:
            best_result = candidate

    experiments_df = pd.DataFrame(experiment_rows)
    return best_result, experiments_df


def print_merged_group_stats(group_time_metrics_df, group_to_members, label):
    print(f"\n=== MERGED GROUP STATS: {label} ===")
    if group_time_metrics_df.empty:
        print("No merged groups found in evaluation data.")
        return

    for group_id in sorted(group_time_metrics_df["merged_group"].unique()):
        group_df = (
            group_time_metrics_df[group_time_metrics_df["merged_group"] == group_id]
            .sort_values("time_segment")
        )
        members = ",".join(group_to_members.get(group_id, []))
        print(f"\nMERGED GROUP: {group_id} (members: {members})")
        print(group_df.to_string(index=False))


def main():
    engine = build_engine()
    ## load orders
    df = load_orders(
        engine,
        limit=1_000_000
    )
    ##load work hours for locations
    work_hours_df = load_work_hours_df(engine)

    ## prepare data, validate orders
    df = prepare_data(df)
    location_to_group, group_to_members = build_merged_location_groups()

    grouped_df_for_segments = apply_location_grouping(df, location_to_group)

    ## add feature to order, location segment
    location_segments = build_location_segments(grouped_df_for_segments)

    ## load locations
    locations_df = load_locations(engine)

    ## group orders by hour and add feature orders_count per hour
    hourly_df = build_hourly_dataset(df)

    ## create index on (location_id, hour) and sort, add missing hours
    hourly_df = expand_hourly_grid(hourly_df)

    ## add time featuters as hour, day of week and other
    hourly_df = add_time_features(hourly_df)

    ## add is open flag
    hourly_df = add_is_open_flag(hourly_df, work_hours_df)

    ## add time segment
    hourly_df = add_time_segment(hourly_df, work_hours_df)

    ## merge predefined restaurant pairs into single grouped locations
    hourly_df = apply_location_grouping(hourly_df, location_to_group)
    hourly_df = aggregate_hourly_after_grouping(hourly_df)

    ## group by location segment
    hourly_df = hourly_df.merge(
        location_segments,
        on=LOCATION_COLUMN,
        how="left"
    )

    ## build dataset grouped by (location, date, time segment, location segment)
    segment_df = build_segment_dataset(hourly_df)

    ## fill missing values. In result we get table with (location, date, time segment, location segment, orders, hours, weekday, month, weekend)
    segment_df = expand_segment_grid(segment_df)

    ## add metrics like std, lag, mean and other
    feature_df = create_segment_features(segment_df)

    train_df, test_df = split_train_test(feature_df)

    ## add general location stats
    feature_df = add_location_stats(train_df, feature_df)
    train_df = add_location_stats(train_df, train_df)
    test_df = add_location_stats(train_df, test_df)

    ## get futures
    features, cat_features = build_feature_matrix(feature_df)

    ## train cat boost
    best_result, experiments_df = run_experiments(
        train_df,
        test_df,
        features,
        cat_features
    )
    best_cfg = best_result["config"]
    print(f"\nBEST EXPERIMENT: {best_cfg['name']}")
    print(pd.DataFrame([best_result["experiment_row"]]).to_string(index=False))

    print_segment_time_stats(
        best_result["result_test"],
        best_result["segment_time_test"],
        best_cfg["name"]
    )
    print_top_worst_segment_time(
        best_result["segment_time_test"],
        best_cfg["name"],
        top_n=10
    )
    group_metrics_df, group_time_metrics_df = evaluate_group_predictions(
        best_result["result_test"]
    )
    merged_group_predictions_df = build_merged_group_predictions_df(
        best_result["result_test"],
        group_to_members
    )
    print_merged_group_stats(group_time_metrics_df, group_to_members, best_cfg["name"])
    quantiles_best = compute_segment_time_error_quantiles(best_result["result_test"])
    print_segment_time_quantiles(quantiles_best, best_cfg["name"])

    save_outputs(
        best_result["model"],
        best_result["result_test"],
        best_result["global_test"],
        best_result["segment_test"],
        best_result["segment_time_test"],
        best_result["peak_metrics"],
        best_result["peak_examples"],
        OUTPUT_DIR_LOG,
        experiments_df=experiments_df,
        group_metrics_df=group_metrics_df,
        group_time_metrics_df=group_time_metrics_df,
        merged_group_predictions_df=merged_group_predictions_df
    )


def save_outputs(
    model,
    result_df,
    global_metrics,
    segment_metrics_df,
    segment_time_metrics_df,
    peak_metrics_df,
    peak_examples_df,
    output_dir,
    experiments_df=None,
    group_metrics_df=None,
    group_time_metrics_df=None,
    merged_group_predictions_df=None
):
    os.makedirs(output_dir, exist_ok=True)

    metrics_df = pd.DataFrame([
        {"metric": "mae", "value": global_metrics["mae"]},
        {"metric": "rmse", "value": global_metrics["rmse"]},
        {"metric": "wape", "value": global_metrics.get("wape")},
        {"metric": "mean_abs_pct_error", "value": global_metrics.get("mean_abs_pct_error")},
        {"metric": "rows", "value": global_metrics["rows"]},
    ])

    metrics_df.to_csv(
        os.path.join(output_dir, "global_metrics.csv"),
        index=False
    )

    segment_metrics_df.to_csv(
        os.path.join(output_dir, "segment_metrics.csv"),
        index=False
    )

    segment_time_metrics_df.to_csv(
        os.path.join(output_dir, "segment_time_metrics.csv"),
        index=False
    )
    acceptance_df = (
        segment_time_metrics_df[segment_time_metrics_df["acceptance_applicable"]]
        .sort_values(["acceptance_passed", "mean_abs_pct_error", "mae"], ascending=[True, False, False])
    )
    acceptance_df.to_csv(
        os.path.join(output_dir, "segment_time_acceptance.csv"),
        index=False
    )

    peak_metrics_df.to_csv(
        os.path.join(output_dir, "peak_metrics.csv"),
        index=False
    )

    peak_examples_df.to_csv(
        os.path.join(output_dir, "peak_examples.csv"),
        index=False
    )

    result_df[
        [
            LOCATION_COLUMN,
            "segment_datetime",
            "segment",
            "time_segment",
            "orders_count",
            "prediction",
            "error",
            "abs_error",
        ]
    ].to_csv(
        os.path.join(output_dir, "predictions_full.csv"),
        index=False
    )

    if experiments_df is not None and not experiments_df.empty:
        experiments_df.to_csv(
            os.path.join(output_dir, "experiment_metrics.csv"),
            index=False
        )

    if group_metrics_df is not None and not group_metrics_df.empty:
        group_metrics_df.to_csv(
            os.path.join(output_dir, "merged_group_metrics.csv"),
            index=False
        )
    if group_time_metrics_df is not None and not group_time_metrics_df.empty:
        group_time_metrics_df.to_csv(
            os.path.join(output_dir, "merged_group_time_metrics.csv"),
            index=False
        )
    if (
        merged_group_predictions_df is not None
        and not merged_group_predictions_df.empty
    ):
        merged_group_predictions_df.to_csv(
            os.path.join(output_dir, "merged_group_predictions.csv"),
            index=False
        )

def print_segment_time_quantiles(quantiles_df, label):
    print(f"\n=== SEGMENT+TIME ERROR QUANTILES: {label} ===")
    print(quantiles_df.to_string(index=False))


def print_segment_time_stats(result_df, segment_time_metrics_df, label):
    print(f"\n=== SEGMENT TIME STATS: {label} ===")

    for rest_segment in ["low", "medium", "high", "mega"]:
        seg_time_df = segment_time_metrics_df[
            segment_time_metrics_df["segment"] == rest_segment
        ]
        if seg_time_df.empty:
            continue

        print(f"\nRESTAURANT SEGMENT: {rest_segment}")
        print(
            seg_time_df.sort_values("time_segment")
            .to_string(index=False)
        )

        examples = (
            result_df[result_df["segment"] == rest_segment]
            .sort_values("segment_datetime")
            .head(10)
        )

        print("\nEXAMPLES:")
        print(
            examples[
                [
                    LOCATION_COLUMN,
                    "segment_datetime",
                    "time_segment",
                    "orders_count",
                    "prediction",
                ]
            ]
            .to_string(index=False)
        )


def compute_segment_time_error_quantiles(result_df):
    df = result_df.copy()
    df["abs_error"] = (df["prediction"] - df["orders_count"]).abs()
    denom = df["orders_count"].replace(0, np.nan)
    df["abs_pct_error"] = (df["abs_error"] / denom) * 100
    df["abs_pct_error"] = df["abs_pct_error"].replace([np.inf, -np.inf], np.nan)

    def _agg(g):
        return pd.Series({
            "rows": len(g),
            "abs_error_p50": g["abs_error"].quantile(0.50),
            "abs_error_p90": g["abs_error"].quantile(0.90),
            "abs_error_p95": g["abs_error"].quantile(0.95),
            "abs_error_p99": g["abs_error"].quantile(0.99),
            "abs_pct_error_p50": g["abs_pct_error"].quantile(0.50),
            "abs_pct_error_p90": g["abs_pct_error"].quantile(0.90),
            "abs_pct_error_p95": g["abs_pct_error"].quantile(0.95),
            "abs_pct_error_p99": g["abs_pct_error"].quantile(0.99),
            "mean_abs_error": g["abs_error"].mean(),
            "mean_abs_pct_error": g["abs_pct_error"].mean(),
        })

    quantiles_df = (
        df.groupby(["segment", "time_segment"])
        .apply(_agg)
        .reset_index()
        .sort_values(["segment", "time_segment"])
    )

    return quantiles_df


if __name__ == "__main__":
    main()
