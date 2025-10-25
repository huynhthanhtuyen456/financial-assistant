from datetime import datetime

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
            # response_model=BalanceSheetResponseModel
            )
async def get_balance_sheet(
        symbols: list[str] = Query(default=[]),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        stmt = select(BalanceSheet).where(BalanceSheet.yearly == yearly)
    else:
        stmt = select(BalanceSheet).where(BalanceSheet.symbol.in_(symbols)).where(BalanceSheet.yearly == yearly)

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

    keys = ['cash', 'asset', 'debt', 'equity', 'payable']
    financial_data = {}
    for row in balance_sheet:
        item = row[0].__dict__

        for row in item["balance_sheet"]:
            for key in keys:
                if row["year"] not in financial_data:
                    financial_data[row["year"]] = {row["ticker"]: {}}
                    financial_data[row["year"]][row["ticker"]] = {f"{key}": row[key] or 0}
                else:
                    if key not in financial_data[row["year"]]:
                        if row["ticker"] not in financial_data[row["year"]]:
                            financial_data[row["year"]][row["ticker"]] = {}
                        financial_data[row["year"]][row["ticker"]][key] = row[key] or 0
                # financial_data[row["year"]][key].reverse()
    data = financial_data
    reversed_dict = {k: data[k] for k in sorted(data)}
    return reversed_dict

    # return {
    #     "data": data,
    #     "status": True
    # }


@router.get("/cashflow",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing cash flow data",
            # response_model=CashflowResponseModel
            )
async def get_cashflow(
        symbols: list[str] = Query(default=[]),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    # if len(symbols) == 1:
    #     symbols = [item.strip() for item in symbols[0].split(',')]
    #
    # if not symbols:
    #     return {
    #         "status": False,
    #         "msg": "Invalid symbol",
    #     }
    #
    # stmt = (select(Cashflow)
    #         .where(Cashflow.symbol.in_(symbols))
    #         .where(Cashflow.yearly == yearly))
    # queryset = await session.execute(stmt)
    #
    # try:
    #     cashflow = queryset.fetchall()
    # except NoResultFound:
    #     return {
    #         "status": False,
    #         "msg": "No cashflow found with symbol {}".format(symbols),
    #         "data": [],
    #     }
    #
    # if not cashflow:
    #     return {
    #         "data": [],
    #         "status": False
    #     }
    #
    # data = []
    #
    # for row in cashflow:
    #     item = row[0].__dict__
    #     data.extend(item["cashflow"])
    #
    # return {
    #     "data": data,
    #     "status": True
    # }
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        stmt = select(Cashflow).where(Cashflow.yearly == yearly)
    else:
        stmt = select(Cashflow).where(Cashflow.symbol.in_(symbols)).where(Cashflow.yearly == yearly)

    queryset = await session.execute(stmt)

    try:
        cashflow = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No balance sheet found with symbol {}".format(symbols),
            "data": [],
        }

    if not cashflow:
        return {
            "data": [],
            "status": False
        }

    keys = ['investCost', 'fromInvest', 'fromFinancial', 'fromSale', 'freeCashFlow']
    financial_data = {}
    for row in cashflow:
        item = row[0].__dict__

        for row in item["cashflow"]:
            for key in keys:
                if row["year"] not in financial_data:
                    financial_data[row["year"]] = {row["ticker"]: {}}
                    financial_data[row["year"]][row["ticker"]] = {f"{key}": row[key] or 0}
                else:
                    if key not in financial_data[row["year"]]:
                        if row["ticker"] not in financial_data[row["year"]]:
                            financial_data[row["year"]][row["ticker"]] = {}
                        financial_data[row["year"]][row["ticker"]][key] = row[key] or 0
                # financial_data[row["year"]][key].reverse()
    data = financial_data
    reversed_dict = {k: data[k] for k in sorted(data)}
    return reversed_dict


@router.get("/incomestatement",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing income statement data",
            # response_model=IncomeStatementResponseModel
            )
async def get_income_statement(
        symbols: list[str] = Query(default=[]),
        yearly: bool = Query(...),
        session: AsyncSession = Depends(session_manager.session),
):
    # if len(symbols) == 1:
    #     symbols = [item.strip() for item in symbols[0].split(',')]
    #
    # if not symbols:
    #     return {
    #         "status": False,
    #         "msg": "Invalid symbol",
    #     }
    #
    # stmt = (select(IncomeStatement)
    #         .where(IncomeStatement.symbol.in_(symbols))
    #         .where(IncomeStatement.yearly == yearly))
    # queryset = await session.execute(stmt)
    #
    # try:
    #     income_statement = queryset.fetchall()
    # except NoResultFound:
    #     return {
    #         "status": False,
    #         "msg": "No income statement found with symbol {}".format(symbols),
    #         "data": [],
    #     }
    #
    # if not income_statement:
    #     return {
    #         "data": [],
    #         "status": False
    #     }
    #
    # data = []
    #
    # for row in income_statement:
    #     item = row[0].__dict__
    #     data.extend(item["income_statement"])
    #
    # return {
    #     "data": data,
    #     "status": True
    # }
    if len(symbols) == 1:
        symbols = [item.strip() for item in symbols[0].split(',')]

    if not symbols:
        stmt = select(IncomeStatement).where(IncomeStatement.yearly == yearly)
    else:
        stmt = select(IncomeStatement).where(IncomeStatement.symbol.in_(symbols)).where(IncomeStatement.yearly == yearly)

    queryset = await session.execute(stmt)

    try:
        income_statement = queryset.fetchall()
    except NoResultFound:
        return {
            "status": False,
            "msg": "No balance sheet found with symbol {}".format(symbols),
            "data": [],
        }

    if not income_statement:
        return {
            "data": [],
            "status": False
        }

    keys = ['revenue', 'preTaxProfit', 'postTaxProfit', 'grossProfit', 'investProfit', 'operationIncome']
    financial_data = {}
    for row in income_statement:
        item = row[0].__dict__

        for row in item["income_statement"]:
            for key in keys:
                if row["year"] not in financial_data:
                    financial_data[row["year"]] = {row["ticker"]: {}}
                    financial_data[row["year"]][row["ticker"]] = {f"{key}": row[key] or 0}
                else:
                    if key not in financial_data[row["year"]]:
                        if row["ticker"] not in financial_data[row["year"]]:
                            financial_data[row["year"]][row["ticker"]] = {}
                        financial_data[row["year"]][row["ticker"]][key] = row[key] or 0
                # financial_data[row["year"]][key].reverse()
    data = financial_data
    reversed_dict = {k: data[k] for k in sorted(data)}
    return reversed_dict


@router.get("/financialratio",
            tags=["scfa"],
            response_description="Data will be list of Json objects containing income statement data",
            response_model=FinancialRatioResponseModel)
async def get_financial_ratio(
        symbols: list[str] = Query(default=[]),
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

    # if len(symbols) == 1:
    #     symbols = [item.strip() for item in symbols[0].split(',')]
    #
    # if not symbols:
    #     stmt = select(FinancialRatio).where(FinancialRatio.yearly == yearly)
    # else:
    #     stmt = select(FinancialRatio).where(FinancialRatio.symbol.in_(symbols)).where(FinancialRatio.yearly == yearly)
    #
    # queryset = await session.execute(stmt)
    #
    # try:
    #     financial_ratio = queryset.fetchall()
    # except NoResultFound:
    #     return {
    #         "status": False,
    #         "msg": "No balance sheet found with symbol {}".format(symbols),
    #         "data": [],
    #     }
    #
    # if not financial_ratio:
    #     return {
    #         "data": [],
    #         "status": False
    #     }
    #
    # keys = ['investCost', 'fromInvest', 'fromFinancial', 'fromSale', 'freeCashFlow']
    # financial_data = {}
    # for row in financial_ratio:
    #     item = row[0].__dict__
    #     for row in item["financial_ratio"]:
    #         for key in keys:
    #             if row["year"] not in financial_data:
    #                 financial_data[row["year"]] = {row["ticker"]: {}}
    #                 financial_data[row["year"]][row["ticker"]] = {f"{key}": row[key] or 0}
    #             else:
    #                 if key not in financial_data[row["year"]]:
    #                     if row["ticker"] not in financial_data[row["year"]]:
    #                         financial_data[row["year"]][row["ticker"]] = {}
    #                     financial_data[row["year"]][row["ticker"]][key] = row[key] or 0
    #             # financial_data[row["year"]][key].reverse()
    # data = financial_data
    # reversed_dict = {k: data[k] for k in sorted(data)}
    # return reversed_dict
