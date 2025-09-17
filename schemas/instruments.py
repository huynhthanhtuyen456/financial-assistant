from pydantic import BaseModel


class InstrumentsResponseModel(BaseModel):
    data: list[dict] | None = None
    message: str
    status: bool