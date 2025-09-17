from json import JSONDecodeError

import requests
from fastapi import APIRouter, Query
from pydantic import BaseModel

from schemas.instruments import InstrumentsResponseModel

router = APIRouter(
    prefix="/scta",
)


class IntradayQuotesResponseModel(BaseModel):
    data: list
    message: str
    status: bool


@router.get("/hitprice",
            tags=["scta"],
            response_description="List of hit prices",
            response_model=InstrumentsResponseModel)
async def get_hit_prices(
        symbols: list[str] = Query(...),
):
    instruments_url = f"https://priceapi.bsc.com.vn/datafeed/instruments"

    response = requests.get(
        instruments_url,
        params={"symbols": ",".join(symbols)},
    )
    json_resp = response.json()
    if response.status_code != 200:
        return InstrumentsResponseModel(status=False, message="Invalid Request!", data=None)

    return InstrumentsResponseModel(
        status=True if json_resp["d"] else False,
        message="Fetched list of instruments successfully" if json_resp["d"] else "No data available.",
        data=json_resp["d"]
    )


@router.get("/intraday-quotes",
            tags=["scta"],
            response_description="Intraday Quotes",
            response_model=IntradayQuotesResponseModel)
async def get_intraday_quotes(
        symbol: str = Query(...)
):
    intraday_quotes_url = f"https://svr2.fireant.vn/api/Data/Markets/IntradayQuotes?symbol={symbol}"
    response = requests.get(intraday_quotes_url)

    try:
        if response.status_code != 200:
            raise JSONDecodeError
        json_resp = response.json()
    except JSONDecodeError as e:
        return {
            "status": False,
            "message": f"Invalid Symbol: {symbol}",
            "data": []
        }

    data = []

    for item in json_resp:
        data.append({key.lower(): value for key, value in item.items()})

    if not data:
        return {
            "status": False,
            "message": "No data available",
            "data": data
        }

    return {
        "status": True,
        "message": "success",
        "data": data
    }

