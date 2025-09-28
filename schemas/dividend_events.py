from pydantic import BaseModel


class DividendEventsResponseModel(BaseModel):
    data: list | None = None
    message: str
    status: bool
