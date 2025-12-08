"""
Stock Model Definition.

This module defines the database model for storing general information about
stock entities (companies). It handles localized naming (English/Vietnamese),
listing status, and unique ticker symbols.
"""
from datetime import date

from sqlalchemy import Column, BigInteger
from sqlmodel import Field

from core import models


class Stock(models.TimestampModel, table=True):
    """
    Represents a specific Stock or Company entity in the database.

    This model stores static reference data about a company, including its
    names in different languages, its ticker symbol, and its listing details.
    It inherits from `models.TimestampModel` to track when the record was
    created or updated.

    Attributes:
        id (int): The primary key identifier. Uses BigInteger to ensure
            scalability.
        name (str): The common name of the stock/company.
        eng_name (str): The official name of the company in English.
        vie_name (str): The official name of the company in Vietnamese.
        symbol (str): The unique stock ticker symbol (e.g., 'VNM', 'VIC').
            Must be unique across the database.
        is_listed (bool): A flag indicating if the stock is currently active
            and listed on the exchange.
        listed_date (date): The date the stock was officially listed on the
            exchange.
    """
    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    name: str = Field(index=True, nullable=False)
    eng_name: str = Field(index=True, nullable=False)
    vie_name: str = Field(index=True, nullable=False)
    symbol: str = Field(index=True, nullable=False, unique=True, max_length=30)
    is_listed: bool = Field(nullable=False)
    listed_date: date = Field(nullable=False)
