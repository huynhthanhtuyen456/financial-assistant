from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlmodel.ext.asyncio.session import AsyncSession

from db import session_manager

STOCK_PRICE_TABLE = "stockprice"


RESOLUTION_DICT = {
    # "1": "one_minute_candle",
    # "3": "three_minutes_candle",
    # "5": "fifteen_minutes_candle",
    # "15": "fifteen_minutes_candle",
    # "30": "thirty_minutes_candle",
    # "45": "forty_five_minutes_candle",
    # "1H": "one_hour_candle",
    # "2H": "two_hours_candle",
    # "4H": "four_hours_candle",
    "1D": "one_day_candle",
    "1W": "one_week_candle",
    "1M": "one_month_candle",
    "3M": "three_months_candle",
    "6M": "six_months_candle",
    "1Y": "one_year_candle"
}


router = APIRouter()


@router.get("/stock", tags=["stock"])
async def read_history(
        session: AsyncSession = Depends(session_manager.session),
        start: int = Query(..., alias="from"),
        to: int = Query(...),
        resolution: str = Query(...),
        symbol: str = Query(...),
        countback: Optional[int] = 0
):
    if start > to:
        raise HTTPException(status_code=400, detail={"msg": "start cannot be greater end.", "field": "start"})
    start_datetime = datetime.fromtimestamp(int(start), tz=timezone.utc)
    end_datetime = datetime.fromtimestamp(int(to), tz=timezone.utc)

    if resolution not in RESOLUTION_DICT.keys():
        raise HTTPException(status_code=400, detail={"msg": "Invalid resolution.", "field": "resolution"})

    mat_view = RESOLUTION_DICT.get(resolution)
    if countback > 0:
        countback = f"{countback} days" if countback > 1 else "1 day"
        sql = text(f"""
            SELECT * FROM {mat_view}
            WHERE ts BETWEEN :start_datetime AND :end_datetime 
                AND symbol=:symbol 
                AND ts >= CAST(:end_datetime AS TIMESTAMP) - INTERVAL '{countback}'
            ORDER BY ts ASC;
        """)
    else:
        if countback < 0:
            raise HTTPException(status_code=400, detail={"msg": "Invalid countback.", "field": "countback"})
        sql = text(f"""
            SELECT * FROM {mat_view}
            WHERE ts BETWEEN :start_datetime AND :end_datetime 
                AND symbol=:symbol
            ORDER BY ts ASC;
        """)

    queryset = await session.execute(
        sql,
        params={
            "symbol": symbol,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
        }
    )

    queryset_as_dict = queryset.mappings().all()
    ohlc_response = {
        "t": [],
        "o": [],
        "h": [],
        "l": [],
        "c": [],
        "v": [],
        "symbol": symbol,
        "s": "ok"
    }

    if not queryset_as_dict:
        ohlc_response["s"] = "no_data"
        return ohlc_response

    for row in queryset_as_dict:
        ohlc_response["t"].append(int(datetime.timestamp(row["ts"])))
        ohlc_response["o"].append(row["open"])
        ohlc_response["h"].append(row["high"])
        ohlc_response["l"].append(row["low"])
        ohlc_response["c"].append(row["close"])
        ohlc_response["v"].append(row["volume"])

    return ohlc_response
