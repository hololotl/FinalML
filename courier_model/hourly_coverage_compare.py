"""
Compare planned vs actual courier counts by location and hour.

Standalone script: does not change courier_model/main.py or business_reports.py.

Outputs (readable, compact):
  - location_summary.csv       one row per restaurant
  - location_day_profile.csv   one row per restaurant/day with compact hour profile
  - mismatches.csv             only hours where forecast != actual
  - readable_report.md         short human summary
  - period_week_summary.csv    summary by daypart (morning/lunch/dinner/...)
  - period_location_summary.csv / period_day_profile.csv
  - readable_report_by_period.md
  - by_period/<period>_*.csv   separate files per daypart

Profile format example:
  08:2/3(-1) 09:3/3 13:3/2(+1)
  hour:forecast/actual(diff)  — diff omitted when equal
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import main as cm
    from .partner_grouping import load_partner_grouping
except ImportError:
    import main as cm
    from partner_grouping import load_partner_grouping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SHIFT_PLAN_PATH = (
    SCRIPT_DIR / "res" / "business_reports" / "business_shift_plan.csv"
)
SHIFT_PLAN_PATH = Path(
    os.getenv("HOURLY_COMPARE_SHIFT_PLAN_PATH", DEFAULT_SHIFT_PLAN_PATH)
)
OUTPUT_DIR = Path(
    os.getenv(
        "HOURLY_COMPARE_OUTPUT_DIR",
        SCRIPT_DIR / "res" / "hourly_coverage_compare",
    )
)
MIN_ABS_DIFF_FOR_MISMATCH = int(os.getenv("HOURLY_COMPARE_MIN_ABS_DIFF", "1"))
SPLIT_BY_VEHICLE = os.getenv("HOURLY_COMPARE_SPLIT_BY_VEHICLE", "0") == "1"

# Business-readable dayparts (not model segment labels).
TIME_PERIODS = [
    ("deep_night", "Глубокая ночь", range(0, 6)),
    ("opening", "Открытие", range(6, 8)),
    ("early_morning", "Раннее утро", range(8, 9)),
    ("morning", "Утро", range(9, 11)),
    ("lunch", "Обед", range(11, 15)),
    ("afternoon", "После обеда", range(15, 18)),
    ("dinner", "Ужин", range(18, 21)),
    ("late_evening", "Поздний вечер", range(21, 24)),
]
PERIOD_BY_HOUR = {
    hour: (period_id, period_name)
    for period_id, period_name, hours in TIME_PERIODS
    for hour in hours
}
PERIOD_RU_BY_ID = {period_id: period_name for period_id, period_name, _ in TIME_PERIODS}


def parse_hour(value) -> int:
    text = str(value)
    if ":" in text:
        return int(text.split(":")[0])
    return int(float(text))


def load_shift_plan(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Shift plan not found: {path}")
    plan = pd.read_csv(path)
    required = {
        "location_id",
        "date",
        "vehicle_type",
        "shift_start",
        "shift_finish",
        "slots_to_create",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"Shift plan is missing columns: {missing}")
    plan = plan.copy()
    plan["location_id"] = plan["location_id"].astype(str)
    plan["date"] = pd.to_datetime(plan["date"]).dt.date.astype(str)
    plan["slots_to_create"] = pd.to_numeric(
        plan["slots_to_create"], errors="coerce"
    ).fillna(0).astype(int)
    if "location_name" not in plan.columns:
        plan["location_name"] = plan["location_id"]
    return plan


def build_forecast_hourly(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in plan.itertuples(index=False):
        start = parse_hour(row.shift_start)
        finish = parse_hour(row.shift_finish)
        if finish <= start:
            continue
        for hour in range(start, finish):
            rows.append(
                {
                    "location_id": str(row.location_id),
                    "location_name": str(row.location_name),
                    "date": str(row.date),
                    "hour": hour,
                    "vehicle_type": str(row.vehicle_type),
                    "forecast_couriers": int(row.slots_to_create),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_name",
                "date",
                "hour",
                "vehicle_type",
                "forecast_couriers",
            ]
        )
    hourly = pd.DataFrame(rows)
    return (
        hourly.groupby(
            ["location_id", "location_name", "date", "hour", "vehicle_type"],
            as_index=False,
        )
        .agg(forecast_couriers=("forecast_couriers", "sum"))
    )


def build_actual_hourly(
    schedule_df: pd.DataFrame,
    location_name_map: dict,
    start: pd.Timestamp,
    finish: pd.Timestamp,
) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_name",
                "date",
                "hour",
                "vehicle_type",
                "actual_couriers",
            ]
        )

    rows = []
    for row in schedule_df.itertuples(index=False):
        shift_start = max(row.schedule_start_dt, start)
        shift_finish = min(row.schedule_finish_dt, finish)
        if shift_finish <= shift_start:
            continue
        hour_cursor = shift_start.floor("h")
        while hour_cursor < shift_finish:
            hour_end = hour_cursor + pd.Timedelta(hours=1)
            overlap_start = max(shift_start, hour_cursor)
            overlap_finish = min(shift_finish, hour_end)
            if overlap_finish > overlap_start:
                rows.append(
                    {
                        "location_id": str(getattr(row, cm.LOCATION_COLUMN)),
                        "date": hour_cursor.date().isoformat(),
                        "hour": int(hour_cursor.hour),
                        "vehicle_type": str(row.vehicle_type),
                        "actual_couriers": 1,
                    }
                )
            hour_cursor = hour_end

    if not rows:
        return pd.DataFrame(
            columns=[
                "location_id",
                "location_name",
                "date",
                "hour",
                "vehicle_type",
                "actual_couriers",
            ]
        )

    actual = (
        pd.DataFrame(rows)
        .groupby(["location_id", "date", "hour", "vehicle_type"], as_index=False)
        .agg(actual_couriers=("actual_couriers", "sum"))
    )
    actual["location_name"] = actual["location_id"].map(
        lambda location_id: location_name_map.get(location_id, location_id)
    )
    return actual


def maybe_aggregate_vehicles(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if SPLIT_BY_VEHICLE:
        return df
    group_cols = ["location_id", "location_name", "date", "hour"]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(**{value_col: (value_col, "sum")})
    )
    out["vehicle_type"] = "total"
    return out


def merge_compare(forecast: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    forecast = maybe_aggregate_vehicles(forecast, "forecast_couriers")
    actual = maybe_aggregate_vehicles(actual, "actual_couriers")
    keys = ["location_id", "date", "hour", "vehicle_type"]
    names = (
        pd.concat(
            [
                forecast[["location_id", "location_name"]],
                actual[["location_id", "location_name"]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["location_id"])
        .drop_duplicates("location_id", keep="first")
    )
    merged = forecast.drop(columns=["location_name"]).merge(
        actual.drop(columns=["location_name"]),
        on=keys,
        how="outer",
    )
    merged = merged.merge(names, on="location_id", how="left")
    merged["location_name"] = merged["location_name"].fillna(merged["location_id"])
    merged["forecast_couriers"] = (
        pd.to_numeric(merged["forecast_couriers"], errors="coerce").fillna(0).astype(int)
    )
    merged["actual_couriers"] = (
        pd.to_numeric(merged["actual_couriers"], errors="coerce").fillna(0).astype(int)
    )
    merged["diff"] = merged["forecast_couriers"] - merged["actual_couriers"]
    merged["abs_diff"] = merged["diff"].abs()
    merged = merged[
        (merged["forecast_couriers"] > 0) | (merged["actual_couriers"] > 0)
    ].copy()
    return merged.sort_values(["date", "location_id", "vehicle_type", "hour"])


def format_hour_cell(forecast: int, actual: int) -> str:
    diff = forecast - actual
    if diff == 0:
        return f"{forecast}/{actual}"
    sign = "+" if diff > 0 else ""
    return f"{forecast}/{actual}({sign}{diff})"


def build_day_profiles(compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["location_id", "location_name", "date", "vehicle_type"]
    for key, group in compare.groupby(group_cols, sort=False):
        group = group.sort_values("hour")
        profile = " ".join(
            f"{int(row.hour):02d}:{format_hour_cell(int(row.forecast_couriers), int(row.actual_couriers))}"
            for row in group.itertuples(index=False)
        )
        mismatch_hours = int((group["abs_diff"] >= MIN_ABS_DIFF_FOR_MISMATCH).sum())
        rows.append(
            {
                "location_id": key[0],
                "location_name": key[1],
                "date": key[2],
                "vehicle_type": key[3],
                "active_hours": int(len(group)),
                "mismatch_hours": mismatch_hours,
                "forecast_courier_hours": int(group["forecast_couriers"].sum()),
                "actual_courier_hours": int(group["actual_couriers"].sum()),
                "under_hours": int(group.loc[group["diff"] < 0, "diff"].abs().sum()),
                "over_hours": int(group.loc[group["diff"] > 0, "diff"].sum()),
                "mae": float(group["abs_diff"].mean()) if len(group) else 0.0,
                "exact_match_rate": float((group["diff"] == 0).mean()) if len(group) else 0.0,
                "profile_forecast_over_actual": profile,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mismatch_hours", "mae", "location_name", "date"],
        ascending=[False, False, True, True],
    )


def build_location_summary(compare: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["location_id", "location_name", "vehicle_type"]
    rows = []
    for key, group in compare.groupby(group_cols, sort=False):
        exact = (group["diff"] == 0).sum()
        rows.append(
            {
                "location_id": key[0],
                "location_name": key[1],
                "vehicle_type": key[2],
                "active_hours": int(len(group)),
                "exact_hours": int(exact),
                "exact_match_rate": float(exact / len(group)) if len(group) else 0.0,
                "forecast_courier_hours": int(group["forecast_couriers"].sum()),
                "actual_courier_hours": int(group["actual_couriers"].sum()),
                "hour_diff_total": int(group["diff"].sum()),
                "under_hours": int(group.loc[group["diff"] < 0, "diff"].abs().sum()),
                "over_hours": int(group.loc[group["diff"] > 0, "diff"].sum()),
                "mae": float(group["abs_diff"].mean()),
                "worst_hour_abs_diff": int(group["abs_diff"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae", "under_hours", "over_hours"],
        ascending=[False, False, False],
    )


def build_mismatches(compare: pd.DataFrame) -> pd.DataFrame:
    mismatches = compare[compare["abs_diff"] >= MIN_ABS_DIFF_FOR_MISMATCH].copy()
    mismatches["hour_label"] = mismatches["hour"].map(lambda hour: f"{int(hour):02d}:00")
    return (
        mismatches[
            [
                "location_id",
                "location_name",
                "date",
                "hour_label",
                "vehicle_type",
                "forecast_couriers",
                "actual_couriers",
                "diff",
                "abs_diff",
            ]
        ]
        .sort_values(
            ["abs_diff", "date", "location_name", "hour_label"],
            ascending=[False, True, True, True],
        )
        .drop(columns=["abs_diff"])
    )


def write_readable_report(
    output_path: Path,
    summary: pd.DataFrame,
    profiles: pd.DataFrame,
    compare: pd.DataFrame,
):
    total_hours = len(compare)
    exact = int((compare["diff"] == 0).sum())
    lines = [
        "# Сравнение курьеров по часам: прогноз vs факт",
        "",
        f"- Активных часов (есть прогноз или факт): **{total_hours}**",
        f"- Точных совпадений: **{exact}** ({exact / total_hours:.1%})"
        if total_hours
        else "- Нет данных",
        f"- Курьеро-часы прогноз: **{int(compare['forecast_couriers'].sum())}**",
        f"- Курьеро-часы факт: **{int(compare['actual_couriers'].sum())}**",
        f"- Недобор часов: **{int(compare.loc[compare['diff'] < 0, 'diff'].abs().sum())}**",
        f"- Перебор часов: **{int(compare.loc[compare['diff'] > 0, 'diff'].sum())}**",
        "",
        "## Хуже всего по MAE (топ-15 ресторанов)",
        "",
        "| Ресторан | MAE | Недобор | Перебор | Точность | Прогноз ч | Факт ч |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.head(15).itertuples(index=False):
        lines.append(
            f"| {row.location_name} | {row.mae:.2f} | {row.under_hours} | "
            f"{row.over_hours} | {row.exact_match_rate:.0%} | "
            f"{row.forecast_courier_hours} | {row.actual_courier_hours} |"
        )

    lines.extend(
        [
            "",
            "## Примеры дневных профилей (час:прогноз/факт)",
            "",
            "Формат: `13:3/2(+1)` = в 13:00 прогноз 3, факт 2, перебор +1.",
            "",
        ]
    )
    for row in profiles.head(12).itertuples(index=False):
        lines.append(
            f"### {row.location_name} · {row.date} · {row.vehicle_type}"
        )
        lines.append("")
        lines.append(f"`{row.profile_forecast_over_actual}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def attach_time_periods(compare: pd.DataFrame) -> pd.DataFrame:
    result = compare.copy()
    mapped = result["hour"].map(PERIOD_BY_HOUR)
    result["time_period"] = mapped.map(lambda value: value[0])
    result["time_period_ru"] = mapped.map(lambda value: value[1])
    return result


def build_period_location_summary(compare: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "location_id",
        "location_name",
        "vehicle_type",
        "time_period",
        "time_period_ru",
    ]
    rows = []
    for key, group in compare.groupby(group_cols, sort=False):
        exact = int((group["diff"] == 0).sum())
        rows.append(
            {
                "location_id": key[0],
                "location_name": key[1],
                "vehicle_type": key[2],
                "time_period": key[3],
                "time_period_ru": key[4],
                "active_hours": int(len(group)),
                "exact_hours": exact,
                "exact_match_rate": float(exact / len(group)) if len(group) else 0.0,
                "forecast_courier_hours": int(group["forecast_couriers"].sum()),
                "actual_courier_hours": int(group["actual_couriers"].sum()),
                "avg_forecast_couriers": float(group["forecast_couriers"].mean()),
                "avg_actual_couriers": float(group["actual_couriers"].mean()),
                "under_hours": int(group.loc[group["diff"] < 0, "diff"].abs().sum()),
                "over_hours": int(group.loc[group["diff"] > 0, "diff"].sum()),
                "mae": float(group["abs_diff"].mean()),
                "worst_hour_abs_diff": int(group["abs_diff"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["time_period", "mae", "under_hours", "over_hours"],
        ascending=[True, False, False, False],
    )


def build_period_day_profiles(compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "location_id",
        "location_name",
        "date",
        "vehicle_type",
        "time_period",
        "time_period_ru",
    ]
    for key, group in compare.groupby(group_cols, sort=False):
        group = group.sort_values("hour")
        profile = " ".join(
            f"{int(row.hour):02d}:{format_hour_cell(int(row.forecast_couriers), int(row.actual_couriers))}"
            for row in group.itertuples(index=False)
        )
        rows.append(
            {
                "location_id": key[0],
                "location_name": key[1],
                "date": key[2],
                "vehicle_type": key[3],
                "time_period": key[4],
                "time_period_ru": key[5],
                "active_hours": int(len(group)),
                "mismatch_hours": int(
                    (group["abs_diff"] >= MIN_ABS_DIFF_FOR_MISMATCH).sum()
                ),
                "forecast_courier_hours": int(group["forecast_couriers"].sum()),
                "actual_courier_hours": int(group["actual_couriers"].sum()),
                "avg_forecast_couriers": float(group["forecast_couriers"].mean()),
                "avg_actual_couriers": float(group["actual_couriers"].mean()),
                "under_hours": int(group.loc[group["diff"] < 0, "diff"].abs().sum()),
                "over_hours": int(group.loc[group["diff"] > 0, "diff"].sum()),
                "mae": float(group["abs_diff"].mean()) if len(group) else 0.0,
                "exact_match_rate": float((group["diff"] == 0).mean()) if len(group) else 0.0,
                "profile_forecast_over_actual": profile,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["time_period", "mismatch_hours", "mae", "location_name", "date"],
        ascending=[True, False, False, True, True],
    )


def build_period_week_summary(compare: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period_id, period_name, _ in TIME_PERIODS:
        group = compare[compare["time_period"] == period_id]
        if group.empty:
            continue
        exact = int((group["diff"] == 0).sum())
        rows.append(
            {
                "time_period": period_id,
                "time_period_ru": period_name,
                "active_hours": int(len(group)),
                "exact_hours": exact,
                "exact_match_rate": float(exact / len(group)),
                "forecast_courier_hours": int(group["forecast_couriers"].sum()),
                "actual_courier_hours": int(group["actual_couriers"].sum()),
                "under_hours": int(group.loc[group["diff"] < 0, "diff"].abs().sum()),
                "over_hours": int(group.loc[group["diff"] > 0, "diff"].sum()),
                "mae": float(group["abs_diff"].mean()),
            }
        )
    return pd.DataFrame(rows)


def write_period_readable_report(
    output_path: Path,
    period_week: pd.DataFrame,
    period_summary: pd.DataFrame,
    period_profiles: pd.DataFrame,
):
    lines = [
        "# Сравнение по промежуткам дня: прогноз vs факт",
        "",
        "Промежутки: ночь / открытие / раннее утро / утро / обед / после обеда / ужин / поздний вечер.",
        "",
        "## Сводка по промежуткам",
        "",
        "| Промежуток | Точность | MAE | Недобор | Перебор | Прогноз ч | Факт ч |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in period_week.itertuples(index=False):
        lines.append(
            f"| {row.time_period_ru} | {row.exact_match_rate:.0%} | {row.mae:.2f} | "
            f"{row.under_hours} | {row.over_hours} | "
            f"{row.forecast_courier_hours} | {row.actual_courier_hours} |"
        )

    for period_id, period_name, _ in TIME_PERIODS:
        top = period_summary[period_summary["time_period"] == period_id].head(8)
        if top.empty:
            continue
        lines.extend(
            [
                "",
                f"## {period_name}: хуже всего по MAE",
                "",
                "| Ресторан | MAE | Недобор | Перебор | Ср. прогноз | Ср. факт |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top.itertuples(index=False):
            lines.append(
                f"| {row.location_name} | {row.mae:.2f} | {row.under_hours} | "
                f"{row.over_hours} | {row.avg_forecast_couriers:.1f} | "
                f"{row.avg_actual_couriers:.1f} |"
            )
        sample = period_profiles[period_profiles["time_period"] == period_id].head(3)
        if sample.empty:
            continue
        lines.extend(["", f"### Примеры профилей · {period_name}", ""])
        for row in sample.itertuples(index=False):
            lines.append(
                f"**{row.location_name} · {row.date}**: `{row.profile_forecast_over_actual}`"
            )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_period_files(
    output_dir: Path,
    period_week: pd.DataFrame,
    period_summary: pd.DataFrame,
    period_profiles: pd.DataFrame,
    compare: pd.DataFrame,
):
    period_dir = output_dir / "by_period"
    period_dir.mkdir(parents=True, exist_ok=True)
    period_week.to_csv(output_dir / "period_week_summary.csv", index=False)
    period_summary.to_csv(output_dir / "period_location_summary.csv", index=False)
    period_profiles.to_csv(output_dir / "period_day_profile.csv", index=False)
    write_period_readable_report(
        output_dir / "readable_report_by_period.md",
        period_week,
        period_summary,
        period_profiles,
    )

    for period_id, period_name, _ in TIME_PERIODS:
        summary = period_summary[period_summary["time_period"] == period_id].copy()
        profiles = period_profiles[period_profiles["time_period"] == period_id].copy()
        mismatches = compare[
            (compare["time_period"] == period_id)
            & (compare["abs_diff"] >= MIN_ABS_DIFF_FOR_MISMATCH)
        ].copy()
        if summary.empty and profiles.empty and mismatches.empty:
            continue
        if not mismatches.empty:
            mismatches["hour_label"] = mismatches["hour"].map(
                lambda hour: f"{int(hour):02d}:00"
            )
            mismatches = mismatches[
                [
                    "location_id",
                    "location_name",
                    "date",
                    "hour_label",
                    "time_period_ru",
                    "vehicle_type",
                    "forecast_couriers",
                    "actual_couriers",
                    "diff",
                ]
            ].sort_values(
                ["date", "location_name", "hour_label"]
            )
        summary.to_csv(period_dir / f"{period_id}_location_summary.csv", index=False)
        profiles.to_csv(period_dir / f"{period_id}_day_profile.csv", index=False)
        mismatches.to_csv(period_dir / f"{period_id}_mismatches.csv", index=False)


def main():
    plan = load_shift_plan(SHIFT_PLAN_PATH)
    if plan.empty:
        raise RuntimeError("Shift plan is empty.")

    forecast_hourly = build_forecast_hourly(plan)
    location_name_map = (
        plan.drop_duplicates("location_id")
        .set_index("location_id")["location_name"]
        .astype(str)
        .to_dict()
    )

    dates = pd.to_datetime(plan["date"])
    start = pd.Timestamp(dates.min()).tz_localize("Europe/Moscow")
    finish = (
        pd.Timestamp(dates.max()).tz_localize("Europe/Moscow")
        + pd.Timedelta(days=1)
    )

    partner_grouping = load_partner_grouping()
    engine = cm.build_engine()
    schedule_df = cm.load_schedule_history(
        engine,
        cm.datetime_to_ms(start),
        cm.datetime_to_ms(finish),
    )
    schedule_df = cm.prepare_schedule_history(
        schedule_df,
        partner_grouping.location_to_planning,
    )
    # Keep only planning locations present in the generated plan.
    plan_locations = set(plan["location_id"].astype(str))
    if not schedule_df.empty:
        schedule_df = schedule_df[
            schedule_df[cm.LOCATION_COLUMN].astype(str).isin(plan_locations)
        ].copy()

    actual_hourly = build_actual_hourly(
        schedule_df,
        location_name_map,
        start,
        finish,
    )
    compare = merge_compare(forecast_hourly, actual_hourly)
    compare = attach_time_periods(compare)
    summary = build_location_summary(compare)
    profiles = build_day_profiles(compare)
    mismatches = build_mismatches(compare)
    period_week = build_period_week_summary(compare)
    period_summary = build_period_location_summary(compare)
    period_profiles = build_period_day_profiles(compare)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "location_summary.csv", index=False)
    profiles.to_csv(OUTPUT_DIR / "location_day_profile.csv", index=False)
    mismatches.to_csv(OUTPUT_DIR / "mismatches.csv", index=False)
    compare.to_csv(OUTPUT_DIR / "hourly_compare_full.csv", index=False)
    write_readable_report(
        OUTPUT_DIR / "readable_report.md",
        summary,
        profiles,
        compare,
    )
    save_period_files(
        OUTPUT_DIR,
        period_week,
        period_summary,
        period_profiles,
        compare,
    )

    print(f"Saved hourly coverage compare to: {OUTPUT_DIR}")
    print(
        f"exact={int((compare['diff'] == 0).sum())}/{len(compare)} "
        f"forecast_hours={int(compare['forecast_couriers'].sum())} "
        f"actual_hours={int(compare['actual_couriers'].sum())}"
    )
    print("\nBy time period:")
    print(
        period_week[
            [
                "time_period_ru",
                "exact_match_rate",
                "mae",
                "under_hours",
                "over_hours",
                "forecast_courier_hours",
                "actual_courier_hours",
            ]
        ].to_string(index=False)
    )
    print("\nTop mismatches by MAE:")
    print(
        summary.head(10)[
            [
                "location_name",
                "mae",
                "under_hours",
                "over_hours",
                "exact_match_rate",
                "forecast_courier_hours",
                "actual_courier_hours",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
