# import datetime
# import logging
# import os
# import sys
# import time
#
# import psycopg2
# import pytz
# import requests
# from sqlalchemy import create_engine, text
# from sqlmodel import Session
#
# from config import get_settings
#
# db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
# session = Session(db_engine)
#
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# # Add scheduled tasks, refer to the official documentation: https://apscheduler.readthedocs.io/en/master/
# # use when you want to run the job periodically at certain time(s) of day
# def download_stock_price_daily_task():
#     """
#     Cron job to download data from DNSE on 02:00 every day following Asia/Ho_Chi_Minh timezone will be 19:00 UTC
#     """
#     logger.info('Cron task is run for downloading stock price data from DNSE!')
#     start_datetime = datetime.datetime.now(pytz.utc)
#     start_datetime = start_datetime.replace(
#         hour=8,
#         minute=0,
#         second=0,
#         microsecond=0,
#     )
#     stocks = text("select symbol from stock where symbol is not null")
#     results = session.exec(stocks)
#
#     end_datetime = (start_datetime + datetime.timedelta(hours=12))
#
#     start_datetime = start_datetime.replace(tzinfo=pytz.timezone("Asia/Ho_Chi_Minh"))
#     end_datetime = end_datetime.replace(tzinfo=pytz.timezone("Asia/Ho_Chi_Minh"))
#
#     start_time = start_datetime.timestamp()
#     end_time = end_datetime.timestamp()
#     logger.info(f"Downloading stock price from {start_datetime} to {end_datetime}")
#     for stock in results:
#         symbol = stock._mapping["symbol"]
#         logger.info(f"Start downloading data with Stock: {symbol}"
#                     f" start at {start_datetime.strftime('%Y-%m-%d %H:%M:%S+00')}"
#                     f" end at {end_datetime.strftime('%Y-%m-%d %H:%M:%S+00')}")
#         download_url = (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock?from="
#                         f"{int(start_time)}&to={int(end_time)}&symbol={symbol}&resolution=1D")
#         logger.info(f"Downloading data from {download_url}")
#         response = requests.get(download_url)
#         json_resp = response.json()
#         if response.status_code != 200:
#             logger.error(f"Download error with symbol {symbol}: {json_resp}")
#             continue
#
#         if not json_resp["t"]:
#             logger.warning(f"Download no data with symbol {symbol}: {json_resp}")
#             continue
#
#         data_lst = []
#         for i in range(len(json_resp["t"])):
#             dt = datetime.datetime.fromtimestamp(
#                 json_resp["t"][i],
#                 tz=pytz.UTC
#             )
#             data = (
#                 symbol,
#                 dt.strftime('%Y-%m-%d %H:%M:%S+00'),
#                 json_resp["o"][i],
#                 json_resp["h"][i],
#                 json_resp["l"][i],
#                 json_resp["c"][i],
#                 json_resp["v"][i],
#             ).__repr__()
#             data_lst.append(data)
#             logger.info(data)
#
#         logger.info(f"Import stock price data with symbl {symbol} to database")
#         values = ",".join(tuple(data_lst))
#         stmt = text(f"""
#             INSERT INTO
#             stockprice(symbol, "time", "open", high, low, "close", volume)
#             VALUES {values};
#         """)
#         session.execute(stmt)
#         session.commit()
#         logger.info(f"Imported stock price data with symbl {symbol} to database")
#         time.sleep(2)
#
#     utc_now = datetime.datetime.now(pytz.utc)
#     utc_now = utc_now.replace(tzinfo=pytz.UTC)
#     logger.info(f"Refresh candle stick views for data from {start_datetime} to {end_datetime}.")
#     candle_sticks = [
#         "one_minute_candle",
#         "three_minutes_candle",
#         "five_minutes_candle",
#         "fifteen_minutes_candle",
#         "thirty_minutes_candle",
#         "forty_five_minutes_candle",
#         "one_hour_candle",
#         "two_hours_candle",
#         "four_hours_candle",
#         "one_day_candle",
#         "one_week_candle",
#         "one_month_candle",
#         "three_months_candle",
#         "six_months_candle",
#         "one_year_candle",
#     ]
#     conn = psycopg2.connect(database=os.environ["DB_NAME"],
#                             user=os.environ["DB_USER"],
#                             password=os.environ["DB_PASSWORD"],
#                             host=os.environ["DB_HOST"],
#                             port=os.environ["DB_PORT"],
#                             )
#     conn.autocommit = True
#     cursor = conn.cursor()
#     for candle in candle_sticks:
#         cursor.execute(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
#          '{utc_now.strftime('%Y-%m-%d')}');""")
#
#     cursor.close()
#     conn.close()
#
#
# if __name__ == "__main__":
#     download_stock_price_daily_task()
#
#     logger.info("Finished importing stock price data...")


import csv
import datetime
import logging
import os
import sys

import math
import boto3
import pytz
from botocore.exceptions import ClientError
from sqlalchemy import text
from sqlmodel import Session, create_engine
from botocore.config import Config
import psycopg2

from config import get_settings

_S3_RESOURCE = None  # Global, see _get_s3_resource()
_S3_CLIENT = None  # Global, see _get_s3_client()
DATAFEED_BUCKET_NAME = os.environ['DATAFEED_BUCKET_NAME']
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)


def get_s3_client():
    """Return S3 Client, set global to on first use only.
    For Lambdas, we want to do expensive initialization in the cold-start so
    they'll be ready for subseqent invocations. But making a boto3 call
    directly on the module's global variable when the file is imported makes an
    HTTP request to AWS, and we have not been able to patch() it away. We can
    easily patch("_get_s3_client") for our tests.
    """
    global _S3_CLIENT

    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client(
            's3',
            config=Config(signature_version='s3v4'),
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

    return _S3_CLIENT


def refresh_materialized_view():
    utc_now = datetime.datetime.now(pytz.utc)
    logger.info(f"Refresh candle stick views for data from {utc_now=}.")
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
        print(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
         '{utc_now.strftime('%Y-%m-%d')}');""")
        cursor.execute(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
         '{utc_now.strftime('%Y-%m-%d')}');""")

    cursor.close()
    conn.close()


def import_price_data_from_s3():
    delete_sql_insert = text("TRUNCATE TABLE stockprice;")
    session.execute(delete_sql_insert)
    session.commit()

    stocks = text("select symbol from stock where symbol is not null")
    results = session.exec(stocks)

    for stock in results:
        symbol = stock._mapping["symbol"]
        logger.info(f"Start downloading income statement with Stock yearly: {symbol}")
        s3_client = get_s3_client()
        filename = f"{symbol}.csv"
        try:
            response = s3_client.get_object(Bucket=DATAFEED_BUCKET_NAME, Key=filename)
        except ClientError as e:
            logger.error(f"{e.response['Error']}. Does not found {filename=}!")
            continue
        csv_reader = csv.DictReader(response['Body'].read().decode('utf-8').splitlines())
        data_list = []
        for row in csv_reader:
            data = (
                row["Ticker"],
                row["Date"],
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]) if row["Volume"] else 0,
            ).__repr__()
            logger.debug(f"{data=}")
            data_list.append(data)

        # Used for pagination
        data_length = len(data_list)
        limit = 1500
        total_page = math.ceil(data_length / limit)
        logger.info(f"{filename=} | {data_length=} | {total_page=}")
        for page in range(1, total_page + 1):
            start_offset = (page - 1) * limit
            end_offset = start_offset + limit
            logger.info(f"{filename=} | {start_offset=} | {end_offset=}")
            values = ",".join(tuple(data_list[start_offset:end_offset]))
            if values:
                sql_insert = text(f"""
                INSERT INTO
                stockprice(symbol, "time", "open", high, low, "close", volume)
                VALUES {values};
                """)
                logger.info(sql_insert)
            logger.info(f"Inserted data in {filename=} | {data_length=} | {total_page=}")
            session.execute(sql_insert)
            session.commit()

    session.close()


def import_price_data_from_s3_v2():
    delete_sql_insert = text("TRUNCATE TABLE stockprice;")
    session.execute(delete_sql_insert)
    session.commit()

    continue_token = None
    s3_client = get_s3_client()
    while True:
        if continue_token is None:
            result = s3_client.list_objects_v2(Bucket=DATAFEED_BUCKET_NAME)
        else:
            result = s3_client.list_objects_v2(Bucket=DATAFEED_BUCKET_NAME, ContinuationToken=continue_token)

        if "Contents" in result:
            for key in result["Contents"]:
                keyString = key["Key"]
                logger.info(f"Start downloading income statement with Stock yearly: {keyString}")
                try:
                    response = s3_client.get_object(Bucket=DATAFEED_BUCKET_NAME, Key=keyString)
                except ClientError as e:
                    logger.error(f"{e.response['Error']}. Does not found {keyString=}!")
                    continue
                csv_reader = csv.DictReader(response['Body'].read().decode('utf-8').splitlines())
                data_list = []
                for row in csv_reader:
                    data = (
                        row["Ticker"],
                        row['Date'],
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row["Volume"]) if row["Volume"] else 0,
                    ).__repr__()
                    logger.debug(f"{data=}")
                    data_list.append(data)

                # Used for pagination
                data_length = len(data_list)
                limit = 1500
                total_page = math.ceil(data_length / limit)
                logger.info(f"{keyString=} | {data_length=} | {total_page=}")
                for page in range(1, total_page + 1):
                    start_offset = (page - 1) * limit
                    end_offset = start_offset + limit
                    logger.info(f"{keyString=} | {start_offset=} | {end_offset=}")
                    values = ",".join(tuple(data_list[start_offset:end_offset]))
                    if values:
                        sql_insert = text(f"""
                        INSERT INTO
                        stockprice(symbol, "time", "open", high, low, "close", volume)
                        VALUES {values};
                        """)
                        logger.info(sql_insert)
                    logger.info(f"Inserted data in {keyString=} | {data_length=} | {total_page=}")
                    session.execute(sql_insert)
                    session.commit()

        if not "NextContinuationToken" in result:
            break

        continue_token = result["NextContinuationToken"]

    session.close()

    utc_now = datetime.datetime.now(pytz.utc)
    logger.info(f"Refresh candle stick views.")
    candle_sticks = [
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
        logger.info(f"Start refreshing candle {candle}")
        cursor.execute(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
         '{utc_now.strftime('%Y-%m-%d')}');""")
        logger.info(f"Finished refreshing candle {candle}")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    logger.info("Importing stock price data...")

    # import_price_data_from_s3()
    import_price_data_from_s3_v2()
    refresh_materialized_view()
    logger.info("Finished importing stock price data...")
