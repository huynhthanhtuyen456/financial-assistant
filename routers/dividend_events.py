import json

import pymongo
from bson import ObjectId
from fastapi import APIRouter
from datetime import datetime

from db import collection
from schemas.dividend_events import DividendEventsResponseModel

router = APIRouter()

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


@router.get("/scfa/dividend",
            tags=["devidend"],
            response_description="Dividend events",
            response_model=DividendEventsResponseModel)
async def get_dividend_events(
        symbol: str | None = None
):
    if symbol:
        dividend_events = collection.find({"symbol": symbol}).sort("published_date", pymongo.DESCENDING)
    else:
        dividend_events = (collection.find({"symbol": {"$ne" : None}})
                           .sort("published_date", pymongo.DESCENDING))

    data = []

    for dividend in dividend_events:
        dividend["id"] = str(dividend["_id"])
        dividend["published_date"] = datetime.fromtimestamp(dividend["published_date"]) if dividend["published_date"] else None
        dividend["record_date"] = datetime.fromtimestamp(dividend["record_date"]) if dividend["record_date"] else None
        dividend["exright_date"] = datetime.fromtimestamp(dividend["exright_date"]) if dividend["exright_date"] else None
        del dividend["_id"]
        data.append(dividend)

    return {
        "status": True,
        "message": "success",
        "data": data
    }
