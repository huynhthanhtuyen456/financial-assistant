from pydantic import BaseModel, Json, UUID4, JsonValue


class ChartConfigIn(BaseModel):
    name: str
    content: JsonValue
    symbol: str
    resolution: str


class ChartConfigModel(BaseModel):
    id: int
    name: str
    userId: UUID4
    symbol: str
    content: Json
    resolution: str
    timestamp: int


class ChartConfigModelOut(BaseModel):
    id: int
    name: str
    userId: str
    symbol: str
    content: Json
    resolution: str
    timestamp: int


class ChartConfigOut(BaseModel):
    data: list[ChartConfigModelOut]
    status: str


class ChartConfigPostResponse(BaseModel):
    id: int
    status: str


class StudyTemplateModel(BaseModel):
    name: str


class StudyTemplateOut(BaseModel):
    data: list[StudyTemplateModel]
    status: str


class DeleteChartConfigResponseModel(BaseModel):
    status: str


class DeleteStudyTemplateResponseModel(BaseModel):
    status: str
