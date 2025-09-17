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


def download_financial_ratio_from_tcbs():
    stocks = text("select symbol from stock where symbol is not null")
    results = session.exec(stocks)

    for stock in results:
        symbol = stock._mapping["symbol"]
        logger.info(f"Start downloading financial ratio with Stock yearly: {symbol}")
        # Download TCA data from TCBS yearly
        download_url = (f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
                        f"{symbol}/financialratio?yearly=1&isAll=true")
        response = requests.get(download_url)
        json_resp = response.json()
        logger.debug(f"Financial Ratio Symbol {stock} {json_resp=}")
        time.sleep(1)
        if response.status_code != 200:
            logger.info(f"Download error with symbol {symbol}: {json_resp}")
            continue
        logger.debug(f"{json_resp=}")
        if not json_resp:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        try:
            financial_ratio = json.dumps(json_resp)
        except json.encoder.JSONEncoder as e:
            logger.error(e)
            continue
        finally:
            tuple_data = (
                symbol,
                True,
                financial_ratio
            )
            stmt = text(f"""
                INSERT INTO financialratio(symbol, yearly, financial_ratio) 
                VALUES {tuple_data}
                ON CONFLICT (symbol, yearly) DO UPDATE SET financial_ratio = EXCLUDED.financial_ratio;
            """)
            session.execute(stmt)
            session.commit()

        # Download TCA data from TCBS quarterly
        logger.info(f"Start downloading income statement with Stock quarterly: {symbol}")
        download_url = (f"https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/"
                        f"{symbol}/financialratio?yearly=0&isAll=true")
        response = requests.get(download_url)
        json_resp = response.json()
        logger.debug(f"Financial Ratio Symbol {stock} {json_resp=}")
        time.sleep(1)
        if response.status_code != 200:
            logger.info(f"Download error with symbol {symbol}: {json_resp}")
            continue
        logger.debug(f"{json_resp=}")
        if not json_resp:
            print(f"Download error with symbol {symbol}: {json_resp}")
            continue
        try:
            financial_ratio = json.dumps(json_resp)
        except json.encoder.JSONEncoder as e:
            logger.error(e)
            continue
        finally:
            tuple_data = (
                symbol,
                False,
                financial_ratio
            )
            stmt = text(f"""
                INSERT INTO financialratio(symbol, yearly, financial_ratio) 
                VALUES {tuple_data}
                ON CONFLICT (symbol, yearly) DO UPDATE SET financial_ratio = EXCLUDED.financial_ratio;
            """)
            session.execute(stmt)
            session.commit()


if __name__ == "__main__":
    print("Downloading Financial Ratio data from TCBS...")

    download_financial_ratio_from_tcbs()
    session.close()

    print("Downloaded Financial Ratio data from TCBS")
