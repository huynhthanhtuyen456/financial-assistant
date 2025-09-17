from sqlalchemy import Column, BigInteger, UniqueConstraint
from sqlmodel import Field, JSON

from core import models


class BalanceSheet(models.TimestampModel, table=True):
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "yearly",
            name="symbol_yearly_balance_sheet",
        ),
    )
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    symbol: str = Field(nullable=False, index=True)
    yearly: bool = Field(nullable=False, default=False)
    balance_sheet: str = Field(
        sa_column=Column(
            JSON(),
            nullable=False
        )
    )
