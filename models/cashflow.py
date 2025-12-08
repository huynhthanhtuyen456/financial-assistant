"""
Cash Flow Model Definition.

This module defines the database model for storing financial cash flow statements.
It utilizes SQLModel for Object-Relational Mapping (ORM) and SQLAlchemy for
database-specific column types (BigInteger, JSON) and constraints.
"""
from sqlalchemy import Column, BigInteger, UniqueConstraint
from sqlmodel import Field, JSON

from core import models


class Cashflow(models.TimestampModel, table=True):
    """
    Represents a financial Cash Flow Statement record in the database.

    This model stores cash flow data as a JSON object, linking it to a specific
    stock symbol and a reporting period (yearly or quarterly). It inherits from
    `models.TimestampModel` to automatically track creation and modification times.

    Attributes:
        id (int): The primary key identifier. Uses BigInteger to support a high
            volume of financial records.
        symbol (str): The stock ticker symbol (e.g., 'HPG', 'VNM'). Indexed to
            optimize query performance.
        yearly (bool): A flag indicating the reporting interval. If True, the
            data represents an annual report. If False, it represents a
            quarterly report. Defaults to False.
        cashflow (dict): The raw cash flow statement data stored as a JSON object.

    Table Constraints:
        UniqueConstraint: Enforces a unique combination of `symbol` and `yearly`.
        This prevents duplicate records for the same reporting period type for a
        single company.
    """
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "yearly",
            name="symbol_yearly_cashflow",
        ),
    )
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    symbol: str = Field(nullable=False, index=True)
    yearly: bool = Field(nullable=False, default=False)
    cashflow: str = Field(
        sa_column=Column(
            JSON(),
            nullable=False
        )
    )
