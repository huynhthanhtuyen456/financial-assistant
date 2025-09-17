from sqlmodel import Session, create_engine

from config import get_settings
from models.stock import StockFloor

db_engine = create_engine(get_settings().database_psycopg_url, echo=True)
session = Session(db_engine)


def import_stockfloor():
    hose = StockFloor(name="HOSE")
    upcom = StockFloor(name="UPCOM")
    hnx = StockFloor(name="HNX")
    session.add(hose)
    session.add(upcom)
    session.add(hnx)
    session.commit()


if __name__ == "__main__":
    print("Importing stock floor data...")

    import_stockfloor()

    print("Finished importing stock floor data...")