"""
Balance Sheet Model Definition.

This module defines the database model for storing financial balance sheet information.
It utilizes SQLModel for Object-Relational Mapping (ORM) and SQLAlchemy for specific
column types and constraints.
"""
from sqlalchemy import Column, BigInteger, UniqueConstraint
from sqlmodel import Field, JSON

from core import models


class BalanceSheet(models.TimestampModel, table=True):
    """
    Represents a financial Balance Sheet record in the database.

    This model stores the balance sheet data as a JSON object, associated with
    a specific stock symbol and a time period indicator (yearly vs. quarterly).
    It inherits from `models.TimestampModel` to include creation and update timestamps.

    Attributes:
        id (int): The primary key identifier. Uses BigInteger to accommodate
            large datasets.
        symbol (str): The stock ticker symbol (e.g., 'FPT', 'HPG'). Indexed
            for fast lookups.
        yearly (bool): A flag indicating the reporting period. If True, the data
            represents a yearly report. If False, it typically represents a
            quarterly report. Defaults to False.
        balance_sheet (dict): The raw financial data stored as a JSON object.

    Table Constraints:
        UniqueConstraint: Ensures that a specific `symbol` can only have one
        entry per `yearly` status (e.g., one yearly record and one quarterly
        record set per symbol, depending on how the data ingestion logic is structured).
    """
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
