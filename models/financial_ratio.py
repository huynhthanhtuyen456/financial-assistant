from sqlalchemy import Column, BigInteger, UniqueConstraint
from sqlmodel import Field, JSON

from core import models


class FinancialRatio(models.TimestampModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "yearly",
            name="symbol_yearly_financial_ratio",
        ),
    )
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    symbol: str = Field(nullable=False, index=True)
    yearly: bool = Field(nullable=False, default=False)
    financial_ratio: str = Field(
        sa_column=Column(
            JSON(),
            nullable=False
        )
    )
