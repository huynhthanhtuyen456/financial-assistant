import json
import os

from sqlalchemy import text
from sqlmodel import Session, create_engine

from config import get_settings

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)


def import_stocklist():
    with open(os.path.join(data_dir, "stock_list.json"), 'r') as f:
        data = json.load(f)
        stock_list = data["data"]
        """
        Sample data:
        {
            "code": "E1VFVN30",
            "companyName": "Quỹ ETF DCVFMVN30",
            "floor": "HOSE",
            "shortName": "Quỹ ETF DCVFMVN30",
            "companyNameEng": "DCVFMVN30 ETF"
        }
        """
        data_lst = []
        for stock in stock_list:
            # floor = select(StockFloor).where(StockFloor.name == stock["floor"])
            data = (
                stock["companyName"],
                stock["code"],
                str(stock["companyNameEng"]) if "companyNameEng" in stock else '',
                stock["companyName"],
            ).__repr__()
            data_lst.append(data)
        values = ",".join(tuple(data_lst))
        stmt = text(f"""INSERT INTO stock(name, symbol, eng_name, vie_name) VALUES {values};""")
        session.execute(stmt)
        session.commit()
        session.close()


if __name__ == "__main__":
    print("Importing stock price data...")

    data_dir = os.path.join(os.getcwd())

    print(data_dir)

    import_stocklist()

    print("Finished importing stock price data...")
