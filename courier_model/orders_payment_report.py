import io
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from .partner_grouping import load_partner_grouping, normalize_location_id
except ImportError:
    from partner_grouping import load_partner_grouping, normalize_location_id


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "orders_payment_report.csv"
OUTPUT_PATH = Path(os.getenv("ORDERS_PAYMENT_REPORT_PATH", DEFAULT_OUTPUT_PATH))
DB_URL = os.getenv(
    "DB_URL",
    "postgresql://courier:1337@localhost:1338/coffee",
)
REPORT_START = os.getenv("PAYMENT_REPORT_START", "").strip()
REPORT_FINISH = os.getenv("PAYMENT_REPORT_FINISH", "").strip()

DATA_COLUMNS = [
    "location_id",
    "location_name",
    "hour_from",
    "hour_to",
    "orders_count",
    "avg_c_rate_total_rub",
    "sum_c_rate_total_rub",
]


def _default_period():
    finish = pd.Timestamp.now(tz="Europe/Moscow").floor("D")
    return finish - pd.Timedelta(days=30), finish


def _parse_period():
    default_start, default_finish = _default_period()
    start = pd.Timestamp(REPORT_START) if REPORT_START else default_start
    finish = pd.Timestamp(REPORT_FINISH) if REPORT_FINISH else default_finish
    if start.tzinfo is None:
        start = start.tz_localize("Europe/Moscow")
    if finish.tzinfo is None:
        finish = finish.tz_localize("Europe/Moscow")
    if finish <= start:
        raise ValueError("PAYMENT_REPORT_FINISH must be after PAYMENT_REPORT_START")
    return start, finish


def _timestamp_to_ms(value):
    return int(pd.Timestamp(value).tz_convert("UTC").timestamp() * 1000)


def load_payment_rows(engine, start, finish):
    query = text(
        """
        SELECT
            o.location_id,
            l.name AS location_name,
            EXTRACT(
                HOUR FROM to_timestamp(o.delivering_at / 1000.0)
                AT TIME ZONE 'Europe/Moscow'
            )::int AS hour_from,
            COUNT(*)::int AS orders_count,
            (AVG(o.c_rate_total) / 100.0)::float AS avg_c_rate_total_rub,
            (SUM(o.c_rate_total) / 100.0)::float AS sum_c_rate_total_rub
        FROM orders o
        LEFT JOIN locations l ON l.id = o.location_id
        WHERE o.delivering_at >= :start_ms
          AND o.delivering_at < :finish_ms
          AND o.c_rate_total IS NOT NULL
          AND o.c_rate_total > 0
        GROUP BY o.location_id, l.name, hour_from
        ORDER BY o.location_id, hour_from
        """
    )
    rows = pd.read_sql_query(
        query,
        engine,
        params={
            "start_ms": _timestamp_to_ms(start),
            "finish_ms": _timestamp_to_ms(finish),
        },
    )
    rows["hour_to"] = (rows["hour_from"] + 1) % 24
    return rows[DATA_COLUMNS]


def save_payment_report(rows, path, start, finish):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        ("period_from", pd.Timestamp(start).date().isoformat()),
        ("period_to", pd.Timestamp(finish).date().isoformat()),
        ("total_orders", int(rows["orders_count"].sum())),
        (
            "total_c_rate_total_rub",
            f"{rows['sum_c_rate_total_rub'].sum():.2f}",
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        for key, value in metadata:
            output.write(f"{key},{value}\n")
        output.write("\n")
        rows.to_csv(output, index=False)


def load_orders_payment_report(path=OUTPUT_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Orders payment report not found: {path}")

    metadata = {}
    data_lines = []
    header_found = False
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("location_id,"):
                header_found = True
            if header_found:
                data_lines.append(line)
            elif line.strip() and "," in line:
                key, value = line.rstrip("\n").split(",", 1)
                metadata[key.strip()] = value.strip()

    if not data_lines:
        raise ValueError(f"Payment report has no data header: {path}")
    rows = pd.read_csv(io.StringIO("".join(data_lines)), dtype={"location_id": str})
    missing = sorted(set(DATA_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Payment report is missing columns: {missing}")
    rows = rows[DATA_COLUMNS].copy()
    rows["location_id"] = rows["location_id"].map(normalize_location_id)
    numeric_columns = [
        "hour_from",
        "hour_to",
        "orders_count",
        "avg_c_rate_total_rub",
        "sum_c_rate_total_rub",
    ]
    rows[numeric_columns] = rows[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    rows = rows.dropna(
        subset=[
            "location_id",
            "hour_from",
            "orders_count",
            "sum_c_rate_total_rub",
        ]
    )
    return rows, metadata


def aggregate_payment_rates(rows, grouping):
    result = rows.copy()
    result["planning_location_id"] = result["location_id"].map(
        grouping.planning_location_id
    )
    grouped = (
        result.groupby(["planning_location_id", "hour_from"], as_index=False)
        .agg(
            orders_count=("orders_count", "sum"),
            sum_c_rate_total_rub=("sum_c_rate_total_rub", "sum"),
        )
    )
    grouped["avg_c_rate_total_rub"] = (
        grouped["sum_c_rate_total_rub"]
        / grouped["orders_count"].replace(0, pd.NA)
    )
    return grouped


def build_payment_quality_audit(rows, metadata, forecast_start=None):
    period_to = pd.to_datetime(metadata.get("period_to"), errors="coerce")
    forecast_ts = pd.to_datetime(forecast_start, errors="coerce")
    future_leakage = bool(
        pd.notna(period_to)
        and pd.notna(forecast_ts)
        and period_to.date() > forecast_ts.date()
    )
    return pd.DataFrame(
        [
            {"metric": "report_rows", "value": len(rows)},
            {
                "metric": "report_locations",
                "value": rows["location_id"].nunique(),
            },
            {
                "metric": "report_orders",
                "value": int(rows["orders_count"].sum()),
            },
            {"metric": "period_from", "value": metadata.get("period_from", "")},
            {"metric": "period_to", "value": metadata.get("period_to", "")},
            {"metric": "future_data_leakage", "value": future_leakage},
        ]
    )


def main():
    start, finish = _parse_period()
    rows = load_payment_rows(create_engine(DB_URL), start, finish)
    if rows.empty:
        raise RuntimeError("No paid orders found for the selected period.")
    save_payment_report(rows, OUTPUT_PATH, start, finish)
    grouping = load_partner_grouping()
    planning_rows = aggregate_payment_rates(rows, grouping)
    planning_path = OUTPUT_PATH.with_name("orders_payment_report_planning.csv")
    planning_rows.to_csv(planning_path, index=False)
    print(f"Saved payment report to: {OUTPUT_PATH}")
    print(f"Saved planning payment rates to: {planning_path}")


if __name__ == "__main__":
    main()
