import csv
import datetime
import logging
import os
import sys

import boto3
import psycopg2
import pytz
from botocore.config import Config
from botocore.exceptions import ClientError
from sqlalchemy import text
from sqlmodel import Session, create_engine

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
                row['Date'],
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]) if row["Volume"] else 0,
            ).__repr__()
            logger.debug(f"{data=}")
            data_list.append(data)

        values = ",".join(tuple(data_list))
        if values:
            sql_insert = text(f"""
            INSERT INTO
            stockprice(symbol, "time", "open", high, low, "close", volume)
            VALUES {values};
            """)
            logger.info(sql_insert)
        logger.info(f"Inserted data in {filename=} | {len(data_list)=}")
        session.execute(sql_insert)
        session.commit()

    session.close()

    utc_now = datetime.datetime.now(pytz.utc)
    logger.info(f"Refresh candle stick views.")
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
        logger.info(f"Start refreshing candle {candle}")
        cursor.execute(f"""CALL refresh_continuous_aggregate('{candle}', NULL,
         '{utc_now.strftime('%Y-%m-%d')}');""")
        logger.info(f"Finished refreshing candle {candle}")

    cursor.close()
    conn.close()


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

                values = ",".join(tuple(data_list))
                if values:
                    sql_insert = text(f"""
                    INSERT INTO
                    stockprice(symbol, "time", "open", high, low, "close", volume)
                    VALUES {values};
                    """)
                    logger.info(sql_insert)
                logger.info(f"Inserted data in {keyString=} | {len(data_list)=}")
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

    logger.info("Finished importing stock price data...")
