import os

import psycopg2


if __name__ == "__main__":
    conn = psycopg2.connect(database=os.environ["DB_NAME"],
                            user=os.environ["DB_USER"],
                            password=os.environ["DB_PASSWORD"],
                            host=os.environ["DB_HOST"],
                            port=os.environ["DB_PORT"],
                            )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f"SELECT add_compression_policy('stockprice', INTERVAL '7 days', true);")
    resolution_list = [
        {"one_day_candle": "1 day"},
        {"one_week_candle": "1 week"},
        {"one_month_candle": "1 month"},
        {"three_months_candle": "3 months"},
        {"six_months_candle": "6 months"},
        {"one_year_candle": "1 year"}
    ]

    resolution_with_sum_volume = [
        "one_week_candle",
        "one_month_candle",
        "three_months_candle",
        "six_months_candle",
        "one_year_candle",
    ]
    for resolution in resolution_list:
        print(f"Processing candlestick for {resolution=}")
        key, value = next(iter(resolution.items()))
        volume_operator = """volume(candlestick_agg("time", volume, volume))"""\
            if key in resolution_with_sum_volume else "LAST(volume, time)"
        try:
            cursor.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS {key}
                WITH (timescaledb.continuous) AS
                    SELECT
                        time_bucket('{value}'::interval, "time") AS ts,
                        symbol,
                        open(candlestick_agg("time", open, volume)),
                        high(candlestick_agg("time", high, volume)),
                        low(candlestick_agg("time", low, volume)),
                        close(candlestick_agg("time", close, volume)),
                        {volume_operator} AS volume
                    FROM stockprice
                    GROUP BY ts, symbol;
            """)
            print(f"Created materialized view candlestick for {key=}")
        except Exception as e:
            print(f"Failed to create materialized view candlestick for {key=}: {e}! Continue to next resolution.")
            continue

    continuous_aggregate_policy = {
        "one_day_candle": {
            "start_offset": "3 days",
            "end_offset": "1 day",
            "schedule_interval": "1 day"
        },
        "one_week_candle": {
            "start_offset": "3 weeks",
            "end_offset": "1 week",
            "schedule_interval": "1 week"
        },
        "one_month_candle": {
            "start_offset": "4 months",
            "end_offset": "1 month",
            "schedule_interval": "1 month"
        },
        "three_months_candle": {
            "start_offset": "15 months",
            "end_offset": "3 months",
            "schedule_interval": "3 months"
        },
        "six_months_candle": {
            "start_offset": "24 months",
            "end_offset": "6 months",
            "schedule_interval": "6 months"
        },
        "one_year_candle": {
            "start_offset": "4 years",
            "end_offset": "1 year",
            "schedule_interval": "1 year"
        },
    }

    for view, policy in continuous_aggregate_policy.items():
        try:
            print(f"Processing policy for {view=}")
            cursor.execute(f"""
                SELECT add_continuous_aggregate_policy('{view}',
                    start_offset => INTERVAL '{policy["start_offset"]}',
                    end_offset => INTERVAL '{policy["end_offset"]}',
                    schedule_interval => INTERVAL '{policy["schedule_interval"]}'
                );
            """)
            print(f"Created aggregate policy for {view=}")
        except Exception as e:
            print(f"Failed to create aggregate policy for {view=}: {e}! Continue to next policy.")

    cursor.close()
    conn.close()