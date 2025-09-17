from pydantic import BaseModel


class DividendEventsResponseModel(BaseModel):
    data: list
    message: str
    status: bool