from getpass import getpass
from itertools import islice

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


TABLES = [
    "locations",
    "couriers",
    "work_hours",
    "orders",
    "courier_shifts",
    "courier_schedule",
    "courier_schedule_locations",
]

password = getpass("Пароль локального PostgreSQL: ")

connection = {
    "host": "127.0.0.1",
    "port": 1338,
    "user": "courier",
    "password": password,
}

source = psycopg2.connect(dbname="coffee_import", **connection)
target = psycopg2.connect(dbname="coffee", **connection)


def columns(conn, table):
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND is_generated = 'NEVER'
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [row[0] for row in cursor.fetchall()]


INCREMENTAL_COLUMNS = {
    "locations": "id",
    "couriers": "id",
    "orders": "id",
    "courier_shifts": "id",
    "courier_schedule": "id",
    "courier_schedule_locations": "schedule_id",
}


def target_max_value(table, column):
    with target.cursor() as cursor:
        cursor.execute(
            sql.SQL("SELECT COALESCE(MAX({}), 0) FROM {}").format(
                sql.Identifier(column),
                sql.Identifier("public", table),
            )
        )
        return cursor.fetchone()[0]


orders_triggers_disabled = False
try:
    for table in TABLES:
        source_columns = columns(source, table)
        target_columns = set(columns(target, table))
        common_columns = [
            column for column in source_columns if column in target_columns
        ]
        if not common_columns:
            print(f"{table}: нет общих колонок, пропускаем")
            continue

        print(f"{table}: общих колонок {len(common_columns)}")
        incremental_column = INCREMENTAL_COLUMNS.get(table)
        query_arguments = ()
        if incremental_column and incremental_column in common_columns:
            last_value = target_max_value(table, incremental_column)
            comparison = (
                sql.SQL(">=")
                if table == "courier_schedule_locations"
                else sql.SQL(">")
            )
            select_query = sql.SQL(
                "SELECT {} FROM {} WHERE {} {} %s ORDER BY {}"
            ).format(
                sql.SQL(", ").join(map(sql.Identifier, common_columns)),
                sql.Identifier("public", table),
                sql.Identifier(incremental_column),
                comparison,
                sql.Identifier(incremental_column),
            )
            query_arguments = (last_value,)
            print(
                f"{table}: читаем только записи после "
                f"{incremental_column}={last_value}"
            )
        else:
            select_query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(", ").join(map(sql.Identifier, common_columns)),
                sql.Identifier("public", table),
            )

        insert_query = sql.SQL(
            "INSERT INTO {} ({}) VALUES %s ON CONFLICT DO NOTHING"
        ).format(
            sql.Identifier("public", table),
            sql.SQL(", ").join(map(sql.Identifier, common_columns)),
        )

        if table == "orders":
            with target.cursor() as cursor:
                cursor.execute(
                    "ALTER TABLE public.orders DISABLE TRIGGER ALL"
                )
            target.commit()
            orders_triggers_disabled = True

        source_cursor = source.cursor(name=f"read_{table}")
        source_cursor.itersize = 5000
        source_cursor.execute(select_query, query_arguments)

        inserted = 0
        processed = 0
        while True:
            batch = list(islice(source_cursor, 5000))
            if not batch:
                break

            with target.cursor() as target_cursor:
                execute_values(
                    target_cursor,
                    insert_query.as_string(target),
                    batch,
                    page_size=1000,
                )
                inserted += max(target_cursor.rowcount, 0)
            target.commit()
            processed += len(batch)
            print(
                f"{table}: обработано {processed}, добавлено {inserted}",
                flush=True,
            )

        source_cursor.close()
        print(f"{table}: добавлено {inserted}")
finally:
    if orders_triggers_disabled:
        target.rollback()
        with target.cursor() as cursor:
            cursor.execute("ALTER TABLE public.orders ENABLE TRIGGER ALL")
        target.commit()
        print("orders: триггеры включены обратно")
    source.close()
    target.close()