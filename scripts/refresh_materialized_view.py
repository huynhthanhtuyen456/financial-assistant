import datetime
import logging
import os
import sys

import psycopg2
import pytz
from sqlalchemy import create_engine
from sqlmodel import Session

from config import get_settings

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)


# Add scheduled tasks, refer to the official documentation: https://apscheduler.readthedocs.io/en/master/
# use when you want to run the job periodically at certain time(s) of day
def refresh_materialized_view():
    """
    Cron job to download data from DNSE on 02:00 every day following Asia/Ho_Chi_Minh timezone will be 19:00 UTC
    """
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


if __name__ == "__main__":
    refresh_materialized_view()

    logger.info("Finished refreshing materialized view price data...")
