import datetime
import os
import time

import psycopg2
import pytz
import requests
from sqlalchemy import text
from sqlmodel import Session, create_engine

from config import get_settings

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)


def download_stockprice_from_dnse():
    start_datetime = datetime.datetime(
        2024,
        10,
        2,
        8,
        0,
        0,
        0,
        pytz.timezone("Asia/Ho_Chi_Minh")
    )
    utc_now = datetime.datetime.now(pytz.utc)
    today = datetime.datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
    delta = (today - start_datetime).days
    stocks = text("select symbol from stock where symbol is not null")
    results = session.exec(stocks)

    end_datetime = (start_datetime + datetime.timedelta(days=delta))

    start_time = start_datetime.timestamp()
    end_time = end_datetime.timestamp()
    for stock in results:
        symbol = stock._mapping["symbol"]
        print(f"Start downloading data with Stock: {symbol}"
              f" start at {start_datetime.strftime('%Y-%m-%d %H:%M:%S+00')}"
              f" end at {end_datetime.strftime('%Y-%m-%d %H:%M:%S+00')}")
        download_url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from="
                        f"{int(start_time)}&to={int(end_time)}&symbol={symbol}&resolution=1D")
        response = requests.get(download_url)
        json_resp = response.json()
        if response.status_code != 200:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        if not json_resp["t"]:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        data_lst = []
        for i in range(len(json_resp["t"])):
            dt = datetime.datetime.fromtimestamp(
                json_resp["t"][i],
                tz=pytz.UTC
            )
            data = (
                symbol,
                dt.strftime('%Y-%m-%d %H:%M:%S+00'),
                json_resp["o"][i],
                json_resp["h"][i],
                json_resp["l"][i],
                json_resp["c"][i],
                json_resp["v"][i],
            ).__repr__()
            data_lst.append(data)
            print(data)
        print(f"Import stock price data with symbl {symbol} to database")
        values = ",".join(tuple(data_lst))
        stmt = text(f"""
            INSERT INTO
            stockprice(symbol, "time", "open", high, low, "close", volume)
            VALUES {values};
        """)
        session.execute(stmt)
        session.commit()
        print(f"Imported stock price data with symbl {symbol} to database")
        time.sleep(0.5)

        print(f"Refresh candle stick views.")
        candle_sticks = [
            "one_minute_candle",
            "three_minutes_candle",
            "five_minutes_candle",
            "fifteen_minutes_candle",
            "thirty_minutes_candle",
            "forty_five_minutes_candle",
            "one_hour_candle",
            "two_hours_candle",
            "four_hours_candle",
            "one_day_candle",
            "one_week_candle",
            "one_month_candle",
            "three_months_candle",
            "six_months_candle",
            "one_year_candle",
        ]
        conn = psycopg2.connect(database=os.environ["DB_NAME"],
                                user=os.environ["DB_USER"],
                                password=os.environ["DB_PASSWORD"],
                                host=os.environ["DB_HOST"],
                                port=os.environ["DB_PORT"],
                                )
        conn.autocommit = True
        cursor = conn.cursor()
        for candle in candle_sticks:
            cursor.execute(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
             '{utc_now.strftime('%Y-%m-%d')}');""")

        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("Downloading stock price data from EntradeX...")

    download_stockprice_from_dnse()
    session.close()

    print("Downloaded stock price data from EntradeX")
