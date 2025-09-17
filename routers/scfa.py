from fastapi import APIRouter, Depends, Query
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import select

from db import session_manager
from models.balancesheet import BalanceSheet
from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.financial_ratio import FinancialRatio
from schemas.tc_analysis import (
    BalanceSheetResponseModel,
    CashflowResponseModel,
    IncomeStatementResponseModel,
    FinancialRatioResponseModel
)

router = APIRouter(
    prefix="/scfa",
)


@router.get("/balancesheet",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing balance sheet data",
            response_model=BalanceSheetResponseModel)
async def get_balance_sheet(
        symbols: list[str] = Query(...),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        return {
            "status": False,
            "msg": "Invalid symbol",
        }

    stmt = (select(BalanceSheet)
            .where(BalanceSheet.symbol.in_(symbols))
            .where(BalanceSheet.yearly == yearly))
    queryset = await session.execute(stmt)

    try:
        balance_sheet = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No balance sheet found with symbol {}".format(symbols),
            "data": [],
        }

    if not balance_sheet:
        return {
            "data": [],
            "status": False
        }

    data = []

    for row in balance_sheet:
        item = row[0].__dict__
        data.extend(item["balance_sheet"])

    return {
        "data": data,
        "status": True
    }


@router.get("/cashflow",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing cash flow data",
            response_model=CashflowResponseModel)
async def get_cashflow(
        symbols: list[str] = Query(...),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        return {
            "status": False,
            "msg": "Invalid symbol",
        }

    stmt = (select(Cashflow)
            .where(Cashflow.symbol.in_(symbols))
            .where(Cashflow.yearly == yearly))
    queryset = await session.execute(stmt)

    try:
        cashflow = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No cashflow found with symbol {}".format(symbols),
            "data": [],
        }

    if not cashflow:
        return {
            "data": [],
            "status": False
        }

    data = []

    for row in cashflow:
        item = row[0].__dict__
        data.extend(item["cashflow"])

    return {
        "data": data,
        "status": True
    }


@router.get("/incomestatement",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing income statement data",
            response_model=IncomeStatementResponseModel)
async def get_income_statement(
        symbols: list[str] = Query(...),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        return {
            "status": False,
            "msg": "Invalid symbol",
        }

    stmt = (select(IncomeStatement)
            .where(IncomeStatement.symbol.in_(symbols))
            .where(IncomeStatement.yearly == yearly))
    queryset = await session.execute(stmt)

    try:
        income_statement = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No income statement found with symbol {}".format(symbols),
            "data": [],
        }

    if not income_statement:
        return {
            "data": [],
            "status": False
        }

    data = []

    for row in income_statement:
        item = row[0].__dict__
        data.extend(item["income_statement"])

    return {
        "data": data,
        "status": True
    }


@router.get("/financialratio",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing income statement data",
            response_model=FinancialRatioResponseModel)
async def get_financial_ratio(
        symbols: list[str] = Query(...),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        return {
            "status": False,
            "msg": "Invalid symbol",
        }

    stmt = (select(FinancialRatio)
            .where(FinancialRatio.symbol.in_(symbols))
            .where(FinancialRatio.yearly == yearly))
    queryset = await session.execute(stmt)

    try:
        financial_ratio = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No financial ratio found with symbol {}".format(symbols),
            "data": [],
        }
    if not financial_ratio:
        return {
            "data": [],
            "status": False
        }

    data = []

    for row in financial_ratio:
        item = row[0].__dict__
        data.extend(item["financial_ratio"])

    return {
        "data": data,
        "status": True
    }
