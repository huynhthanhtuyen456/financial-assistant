from datetime import date, datetime
from typing import Optional

from sqlalchemy import Column, DateTime, TEXT, BigInteger
from sqlalchemy.orm import Mapped, DeclarativeBase
from sqlmodel import Field, Relationship, SQLModel

from core import models


class Industry(models.TimestampModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    child: int = Field(nullable=True, foreign_key="industry.id")
    parent: Optional["Industry"] = Relationship(back_populates="children")
    children: Mapped[list["Industry"]] = Relationship(
        back_populates="parent", sa_relationship_kwargs={
            "lazy": "joined", "join_depth": 2, "remote_side": "industry.id"
        }
    )
    stocks: list["Stock"] = Relationship(back_populates="industry")


class Sector(models.TimestampModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    industry_id: int = Field(foreign_key=f"{Industry.__tablename__}.id", nullable=False)
    stocks: list["Stock"] = Relationship(back_populates="sector")


class StockFloor(models.TimestampModel, table=True):
    """The Floor includes HOSE, UPCOM, HNX where stock listed"""
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    stocks: list["Stock"] = Relationship(back_populates="floor")


class StockType(models.TimestampModel, table=True):
    """Categorize the stock includes STOCK, ETF."""
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    stocks: list["Stock"] = Relationship(back_populates="type")


class StockGroup(models.TimestampModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    stocks: list["Stock"] = Relationship(back_populates="group")


class Stock(models.TimestampModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field(index=True, nullable=False)
    eng_name: str = Field(index=True, nullable=False)
    vie_name: str = Field(index=True, nullable=False)
    symbol: str = Field(index=True, nullable=False, unique=True, max_length=30)

    # logo: str = Field(nullable=False)
    # is_listed: bool = Field(nullable=False)
    # listed_date: date = Field(index=True, nullable=False)
    #
    # industry_id: int = Field(foreign_key=f"{Industry.__tablename__}.id", nullable=False)
    # industry: Industry | None = Relationship(back_populates="stocks")
    #
    # floor_id: int = Field(foreign_key=f"{StockFloor.__tablename__}.id", nullable=False)
    # floor: StockFloor | None = Relationship(back_populates="stocks")
    #
    # type_id: int = Field(foreign_key=f"{StockType.__tablename__}.id", nullable=False)
    # type: StockType | None = Relationship(back_populates="stocks")
    #
    # group_id: int = Field(foreign_key=f"{StockGroup.__tablename__}.id", nullable=False)
    # group: StockGroup | None = Relationship(back_populates="stocks")
    #
    # sector_id: int = Field(foreign_key=f"{Sector.__tablename__}.id", nullable=False)
    # sector: Sector | None = Relationship(back_populates="stocks")
