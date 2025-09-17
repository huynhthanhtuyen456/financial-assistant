from pydantic import BaseModel


class BalanceSheetResponseModel(BaseModel):
    data: list
    status: bool


class CashflowResponseModel(BaseModel):
    data: list
    status: bool


class IncomeStatementResponseModel(BaseModel):
    data: list
    status: bool


class FinancialRatioResponseModel(BaseModel):
    data: list
    status: bool
