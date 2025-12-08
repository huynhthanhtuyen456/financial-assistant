"""
Income Statement Model Definition.

This module defines the database model for storing financial income statements
(Profit and Loss). It utilizes SQLModel for Object-Relational Mapping (ORM) and
SQLAlchemy for database-specific column types (BigInteger, JSON) and constraints.
"""
from sqlalchemy import Column, BigInteger, UniqueConstraint
from sqlmodel import Field, JSON

from core import models


class IncomeStatement(models.TimestampModel, table=True):
    """
    Represents a financial Income Statement (P&L) record in the database.

    This model stores income statement data as a JSON object, linking it to a
    specific stock symbol and a reporting period (yearly or quarterly). It
    inherits from `models.TimestampModel` to automatically track creation and
    modification timestamps.

    Attributes:
        id (int): The primary key identifier. Uses BigInteger to support large
            datasets typical in financial time-series data.
        symbol (str): The stock ticker symbol (e.g., 'FPT', 'HPG'). Indexed
            to optimize query performance during lookups.
        yearly (bool): A flag indicating the reporting interval. If True, the
            data represents an annual report. If False, it represents a
            quarterly report. Defaults to False.
        income_statement (dict): The raw income statement data stored as a JSON
            object.

    Table Constraints:
        UniqueConstraint: Enforces a unique combination of `symbol` and `yearly`.
        This ensures that only one annual and one quarterly record set exists
        per symbol.
    """
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "yearly",
            name="symbol_yearly_income_statement",
        ),
    )
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    symbol: str = Field(nullable=False, index=True)
    yearly: bool = Field(nullable=False, default=False)
    income_statement: str = Field(
        sa_column=Column(
            JSON(),
            nullable=False
        )
    )
