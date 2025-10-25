import json
import logging
import sys
import time

import requests

from datetime import datetime
from sqlalchemy import text
from sqlmodel import Session, create_engine

from config import get_settings

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG if get_settings().debug_logs else logging.INFO)
logger = logging.getLogger(__name__)


def download_symbol_list():
    url = "https://api.dnse.com.vn/market-api/tickers?_end=2042"

    try:
        response = requests.get(url)
        response.raise_for_status()
        json_resp = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch symbol list: {e}")
        return

    symbols_data = json_resp.get("data", [])

    for symbol_info in symbols_data:
        if symbol_info.get("type") == "STOCK" and symbol_info.get("isListed") == True:
            _insert_or_update_stock(session, symbol_info)


def _insert_or_update_stock(db_session, symbol_info):
    """Insert or update a single stock record in the database."""
    stock_data = {
        "name": symbol_info.get("companyName", ""),
        "symbol": symbol_info.get("symbol", ""),
        "eng_name": symbol_info.get("companyNameEng", ""),
        "vie_name": symbol_info.get("companyNameVie", ""),
        "is_listed": symbol_info.get("isListed", ""),
        "listed_date": symbol_info.get("listedDate"),
    }
    stmt = text("""
        INSERT INTO stock(name, symbol, eng_name, vie_name, is_listed, listed_date)
        VALUES (:name, :symbol, :eng_name, :vie_name, :is_listed, :listed_date) ON CONFLICT (symbol) 
        DO
        UPDATE SET
            name = EXCLUDED.name,
            eng_name = EXCLUDED.eng_name,
            vie_name = EXCLUDED.vie_name,
            is_listed = EXCLUDED.is_listed,
            listed_date = EXCLUDED.listed_date
    """)

    try:
        db_session.execute(stmt, stock_data)
        db_session.commit()
    except Exception as e:
        logger.error(f"Failed to insert/update stock {stock_data['symbol']}: {e}")
        db_session.rollback()
        session.execute(stmt)
        session.commit()


if __name__ == "__main__":
    download_symbol_list()