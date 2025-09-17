from sqlalchemy import Column, BigInteger
from sqlmodel import Field, JSON, String

from core import models


class ChartConfig(models.TimestampModel, table=True):
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    name: str = Field(nullable=False)
    userId: str = Field(
        sa_column=Column(
            String(),
            nullable=False
        )
    )
    resolution: str = Field(nullable=False)
    symbol: str = Field(nullable=False)
    content: str = Field(
        sa_column=Column(
            JSON(),
            nullable=False
        )
    )
