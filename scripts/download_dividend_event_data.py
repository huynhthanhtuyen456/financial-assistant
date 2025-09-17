import logging
import sys
import time
from datetime import datetime

import pytz
import requests

from config import get_settings
from db import collection

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)




def convert_timestamp_in_datetime_utc(timestamp_received):
    dt_naive_utc = datetime.utcfromtimestamp(timestamp_received)
    return dt_naive_utc.replace(tzinfo=pytz.utc)


def download_dividend_event_data():
    collection.delete_many({})
    page = 1

    while page:
        if not page:
            break
        # Download TCA data from TCBS yearly
        download_url = (f"https://api-finance-t19.24hmoney.vn/v1/web/announcement/dividend-events"
                        f"?page={page}&per_page=5000&type=all&floor=all")
        response = requests.get(download_url)
        json_resp = response.json()
        logger.debug(f"Dividend events with {page=} {json_resp=}")
        time.sleep(1)
        if response.status_code != 200:
            logger.info(f"Download error with symbol {page=}: {json_resp}")
            continue

        logger.debug(f"{json_resp=}")
        if not json_resp:
            logger.error(f"Download error with symbol {page=}: {json_resp}")
            continue

        if not json_resp["data"]:
            page = None
            continue

        for data in json_resp["data"]:
            dividend = {
                "symbol": data["symbol"],
                "title": data["title"],
                "published_date": data["published_date"],
                "company_name": data["company_name"],
                "type": data["type"],
                "floor": data["floor"],
                "record_date": data["record_date"] if data["record_date"] else None,
                "exright_date": data["exright_date"] if data["exright_date"] else None,
                "payout_date": data["payout_date"] if data["payout_date"] else None,
            }
            inserted_dividend = collection.insert_one(dividend)
            logger.debug(f"Inserted {inserted_dividend=}")

        page += 1


if __name__ == "__main__":
    print("Downloading Dividend event Data...")

    download_dividend_event_data()

    print("Downloaded Dividend event!")
