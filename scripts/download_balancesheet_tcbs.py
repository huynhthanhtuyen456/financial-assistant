import json
import logging
import sys
import time

import requests
from sqlalchemy import text
from sqlmodel import Session, create_engine

from config import get_settings

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)


def download_balancesheet_from_tcbs():
    stocks = text("select symbol from stock where symbol is not null")
    results = session.exec(stocks)

    for stock in results:
        symbol = stock._mapping["symbol"]
        logger.info(f"Start downloading Balance Sheet with Stock yearly: {symbol}")
        # Download TCA data from TCBS yearly
        download_url = (f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
                        f"{symbol}/balancesheet?yearly=1&isAll=true")
        response = requests.get(download_url)
        logger.debug(f"Download URL: {download_url}")
        try:
            json_resp = response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e} with status code {response.status_code}")
        logger.debug(f"Balance Sheet Symbol {stock} {json_resp=}")
        time.sleep(1)
        if response.status_code != 200:
            logger.info(f"Download error with symbol {symbol}: {json_resp}")
            continue
        logger.debug(f"{json_resp=}")
        if not json_resp:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        try:
            income_statement = json.dumps(json_resp)
        except json.encoder.JSONEncoder as e:
            logger.error(e)
            continue
        finally:
            tuple_data = (
                symbol,
                True,
                income_statement
            )
            stmt = text(f"""
                INSERT INTO balancesheet(symbol, yearly, balance_sheet) 
                VALUES {tuple_data}
                ON CONFLICT (symbol, yearly) DO UPDATE SET balance_sheet = EXCLUDED.balance_sheet;
            """)
            session.execute(stmt)
            session.commit()

        # Download TCA data from TCBS quarterly
        logger.info(f"Start downloading Balance Sheet with Stock quarterly: {symbol}")
        download_url = (f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
                        f"{symbol}/balancesheet?yearly=0&isAll=true")
        response = requests.get(download_url)
        json_resp = response.json()
        logger.debug(f"Balance Sheet Symbol {stock} {json_resp=}")
        time.sleep(1)
        if response.status_code != 200:
            logger.info(f"Download error with symbol {symbol}: {json_resp}")
            continue
        logger.debug(f"{json_resp=}")
        if not json_resp:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        try:
            income_statement = json.dumps(json_resp)
        except json.encoder.JSONEncoder as e:
            logger.error(e)
            continue
        finally:
            tuple_data = (
                symbol,
                False,
                income_statement
            )
            stmt = text(f"""
                INSERT INTO balancesheet(symbol, yearly, balance_sheet) 
                VALUES {tuple_data}
                ON CONFLICT (symbol, yearly) DO UPDATE SET balance_sheet = EXCLUDED.balance_sheet;
            """)
            session.execute(stmt)
            session.commit()


if __name__ == "__main__":
    print("Downloading Balance Sheet Technical Analysis Data from TCBS...")

    download_balancesheet_from_tcbs()
    session.close()

    print("Downloaded Balance Sheet Technical Analysis Data from TCBS")
