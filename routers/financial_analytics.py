import base64
import io

import matplotlib
import pandas as pd
import requests

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from db import session_manager
from models.balancesheet import BalanceSheet
from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.financial_ratio import FinancialRatio


router = APIRouter(
    prefix="/financial-analytics",
    tags=["financial-analytics"]
)
templates = Jinja2Templates(directory="templates")

def create_base64_image(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64

@router.get("/dashboard", response_class=HTMLResponse)
async def get_financial_dashboard(
    request: Request,
    symbol: str = Query("FPT", description="Stock symbol to analyze"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get financial analytics dashboard HTML page"""
    symbol = symbol.upper()
    # Fetch data from database
    try:
        # Get balance sheet data
        stmt_bs = (select(BalanceSheet)
                  .where(BalanceSheet.symbol == symbol)
                  .where(BalanceSheet.yearly == yearly))
        queryset_bs = await session.execute(stmt_bs)
        balance_sheet = queryset_bs.fetchall()
        
        if not balance_sheet:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Extract data
        bs_data = []
        for row in balance_sheet:
            item = row[0].__dict__
            bs_data.extend(item["balance_sheet"])
        
        df_balancesheet = pd.DataFrame(bs_data)
        
        # Get cashflow data
        stmt_cf = (select(Cashflow)
                  .where(Cashflow.symbol == symbol)
                  .where(Cashflow.yearly == yearly))
        queryset_cf = await session.execute(stmt_cf)
        cashflow = queryset_cf.fetchall()
        
        cf_data = []
        for row in cashflow:
            item = row[0].__dict__
            cf_data.extend(item["cashflow"])
        
        df_cashflow = pd.DataFrame(cf_data)
        
        # Get income statement data
        stmt_is = (select(IncomeStatement)
                  .where(IncomeStatement.symbol == symbol)
                  .where(IncomeStatement.yearly == yearly))
        queryset_is = await session.execute(stmt_is)
        income_statement = queryset_is.fetchall()
        
        is_data = []
        for row in income_statement:
            item = row[0].__dict__
            is_data.extend(item["income_statement"])
        
        df_incomestatement = pd.DataFrame(is_data)
        
        # Get financial ratio data
        stmt_fr = (select(FinancialRatio)
                  .where(FinancialRatio.symbol == symbol)
                  .where(FinancialRatio.yearly == yearly))
        queryset_fr = await session.execute(stmt_fr)
        financial_ratio = queryset_fr.fetchall()
        
        fr_data = []
        for row in financial_ratio:
            item = row[0].__dict__
            fr_data.extend(item["financial_ratio"])
        
        df_financialratio = pd.DataFrame(fr_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    # Generate visualizations
    visualizations = {}
    
    try:
        # 1. Asset, Debt, Equity trends
        df_bs_sorted = df_balancesheet.sort_values('year')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_bs_sorted['year'], df_bs_sorted['asset'], label='Total Assets', marker='o')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['debt'], label='Total Debt', marker='s')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['equity'], label='Equity', marker='^')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (Billion VND)')
        ax.set_title(f'{symbol} Balance sheet: Assets, Debt, and Equity Over Years')
        ax.legend()
        ax.grid(True)
        visualizations['asset_debt_equity'] = create_base64_image(fig)
        
        # 2. Short and Long Debt trends
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_bs_sorted['year'], df_bs_sorted['longDebt'], label='Long Debt', marker='o')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['shortDebt'], label='Short Debt', marker='s')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (Billion VND)')
        ax.set_title(f'{symbol} Balance sheet: Short and Long Debt Over Years')
        ax.legend()
        ax.grid(True)
        visualizations['debt_breakdown'] = create_base64_image(fig)
        
        # 3. Free Cash Flow trend
        df_cf_sorted = df_cashflow.sort_values('year')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_cf_sorted['year'], df_cf_sorted['freeCashFlow'], marker='o', color='green')
        ax.set_xlabel('Year')
        ax.set_ylabel('Free Cash Flow')
        ax.set_title(f'{symbol} Free Cash Flow Trend')
        ax.grid(True)
        visualizations['free_cashflow'] = create_base64_image(fig)
        
        # 4. Revenue and Net Profit trend
        df_is_sorted = df_incomestatement.sort_values('year')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_is_sorted['year'], df_is_sorted['revenue'], label='Revenue', marker='o')
        ax.plot(df_is_sorted['year'], df_is_sorted['postTaxProfit'], label='Net Profit', marker='s')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(f'{symbol} Revenue and Net Profit Trend')
        ax.legend()
        ax.grid(True)
        visualizations['revenue_profit'] = create_base64_image(fig)
        
        # 5. Correlation heatmap
        df_merged = df_balancesheet[['year', 'asset', 'debt', 'equity']].merge(
            df_cashflow[['year', 'freeCashFlow']], on='year', how='inner'
        ).merge(
            df_incomestatement[['year', 'revenue', 'postTaxProfit']], on='year', how='inner'
        ).merge(
            df_financialratio[['year', 'roe', 'roa']], on='year', how='inner'
        )
        
        corr_matrix = df_merged[['asset', 'debt', 'equity', 'freeCashFlow', 'revenue', 'postTaxProfit', 'roe', 'roa']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title(f'{symbol} Correlation Matrix of Key Financial Features')
        visualizations['correlation_heatmap'] = create_base64_image(fig)
        
        # 6. Revenue prediction model
        features = ['asset', 'debt', 'equity', 'freeCashFlow', 'postTaxProfit', 'roe', 'roa']
        X = df_merged[features]
        y = df_merged['revenue']
        
        # Drop rows with NaN
        mask = X.notnull().all(axis=1) & y.notnull()
        X = X[mask]
        y = y[mask]
        
        if len(X) > 5:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y_test.values, label='Actual Revenue', marker='o')
            ax.plot(y_pred, label='Predicted Revenue', marker='x')
            ax.set_xlabel('Test Sample Index')
            ax.set_ylabel('Revenue')
            ax.set_title(f'{symbol} Actual vs Predicted Revenue (Linear Regression)')
            ax.legend()
            ax.grid(True)
            visualizations['revenue_prediction'] = create_base64_image(fig)
            
            r2_score = lr.score(X_test, y_test)
        else:
            r2_score = None
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context = {
            "symbol": symbol,
            "yearly": yearly,  # or False, for data type selection
            "df_bs_sorted": df_bs_sorted,  # Pandas DataFrame with 'asset', 'debt', 'equity' columns, sorted by date
            "df_is_sorted": df_is_sorted,  # Pandas DataFrame with 'revenue' column, sorted by date
            "visualizations": {
                "asset_debt_equity": visualizations.get('asset_debt_equity', ''),
                "debt_breakdown": visualizations.get('debt_breakdown', ''),
                "free_cashflow": visualizations.get('free_cashflow', ''),
                "revenue_profit": visualizations.get('revenue_profit', ''),
                "correlation_heatmap": visualizations.get('correlation_heatmap', ''),
                "revenue_prediction": visualizations.get('revenue_prediction', ''),
            },
            "r2_score": r2_score,  # or None if not available
        }
    )

@router.get("/chart/balancesheet")
async def get_balancesheet_chart(
    symbol: str = Query("FPT", description="Stock symbol"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get balance sheet visualization as base64 image"""
    try:
        # Fetch balance sheet data
        # stmt = (select(BalanceSheet)
        #         .where(BalanceSheet.symbol == symbol)
        #         .where(BalanceSheet.yearly == yearly))
        # queryset = await session.execute(stmt)
        # balance_sheet = queryset.fetchall()
        
        # if not balance_sheet:
        #     raise HTTPException(status_code=404, detail=f"No balance sheet data found for {symbol}")
        
        # # Extract and process data
        # bs_data = []
        # for row in balance_sheet:
        #     item = row[0].__dict__
        #     bs_data.extend(item["balance_sheet"])
        
        # df_balancesheet = pd.DataFrame(bs_data)
        balancesheet = requests.get(f'https://app.finsc.vn/api/v1/scfa/balancesheet?symbols={symbol}&yearly={yearly}')
        balancesheet = balancesheet.json()
        df_balancesheet = pd.DataFrame(balancesheet['data'])
        df_bs_sorted = df_balancesheet.sort_values('year')
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_bs_sorted['year'], df_bs_sorted['asset'], label='Total Assets', marker='o')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['debt'], label='Total Debt', marker='s')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['equity'], label='Equity', marker='^')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (Billion VND)')
        ax.set_title(f'{symbol} Balancesheet: Assets, Debt, and Equity Over Years')
        ax.legend()
        ax.grid(True)
        
        img_base64 = create_base64_image(fig)
        
        return {
            "status": True,
            "image": img_base64,
            "symbol": symbol,
            "data_points": len(df_bs_sorted)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/chart/cashflow")
async def get_cashflow_chart(
    symbol: str = Query("FPT", description="Stock symbol"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get cash flow visualization as base64 image"""
    try:
        # Fetch cash flow data
        stmt = (select(Cashflow)
                .where(Cashflow.symbol == symbol)
                .where(Cashflow.yearly == yearly))
        queryset = await session.execute(stmt)
        cashflow = queryset.fetchall()
        
        if not cashflow:
            raise HTTPException(status_code=404, detail=f"No cash flow data found for {symbol}")
        
        # Extract and process data
        cf_data = []
        for row in cashflow:
            item = row[0].__dict__
            cf_data.extend(item["cashflow"])
        
        df_cashflow = pd.DataFrame(cf_data)
        df_cf_sorted = df_cashflow.sort_values('year')
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_cf_sorted['year'], df_cf_sorted['freeCashFlow'], marker='o', color='green')
        ax.set_xlabel('Year')
        ax.set_ylabel('Free Cash Flow')
        ax.set_title(f'{symbol} Free Cash Flow Trend')
        ax.grid(True)
        
        img_base64 = create_base64_image(fig)
        
        return {
            "status": True,
            "image": img_base64,
            "symbol": symbol,
            "data_points": len(df_cf_sorted)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/chart/income-statement")
async def get_income_statement_chart(
    symbol: str = Query("FPT", description="Stock symbol"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get income statement visualization as base64 image"""
    try:
        # Fetch income statement data
        stmt = (select(IncomeStatement)
                .where(IncomeStatement.symbol == symbol)
                .where(IncomeStatement.yearly == yearly))
        queryset = await session.execute(stmt)
        income_statement = queryset.fetchall()
        
        if not income_statement:
            raise HTTPException(status_code=404, detail=f"No income statement data found for {symbol}")
        
        # Extract and process data
        is_data = []
        for row in income_statement:
            item = row[0].__dict__
            is_data.extend(item["income_statement"])
        
        df_incomestatement = pd.DataFrame(is_data)
        df_is_sorted = df_incomestatement.sort_values('year')
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_is_sorted['year'], df_is_sorted['revenue'], label='Revenue', marker='o')
        ax.plot(df_is_sorted['year'], df_is_sorted['postTaxProfit'], label='Net Profit', marker='s')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(f'{symbol} Revenue and Net Profit Trend')
        ax.legend()
        ax.grid(True)
        
        img_base64 = create_base64_image(fig)
        
        return {
            "status": True,
            "image": img_base64,
            "symbol": symbol,
            "data_points": len(df_is_sorted)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/chart/correlation")
async def get_correlation_chart(
    symbol: str = Query("FPT", description="Stock symbol"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get correlation heatmap as base64 image"""
    try:
        # Fetch all data
        stmt_bs = (select(BalanceSheet)
                  .where(BalanceSheet.symbol == symbol)
                  .where(BalanceSheet.yearly == yearly))
        queryset_bs = await session.execute(stmt_bs)
        balance_sheet = queryset_bs.fetchall()
        
        stmt_cf = (select(Cashflow)
                  .where(Cashflow.symbol == symbol)
                  .where(Cashflow.yearly == yearly))
        queryset_cf = await session.execute(stmt_cf)
        cashflow = queryset_cf.fetchall()
        
        stmt_is = (select(IncomeStatement)
                  .where(IncomeStatement.symbol == symbol)
                  .where(IncomeStatement.yearly == yearly))
        queryset_is = await session.execute(stmt_is)
        income_statement = queryset_is.fetchall()
        
        stmt_fr = (select(FinancialRatio)
                  .where(FinancialRatio.symbol == symbol)
                  .where(FinancialRatio.yearly == yearly))
        queryset_fr = await session.execute(stmt_fr)
        financial_ratio = queryset_fr.fetchall()
        
        if not all([balance_sheet, cashflow, income_statement, financial_ratio]):
            raise HTTPException(status_code=404, detail=f"Incomplete data found for {symbol}")
        
        # Extract data
        bs_data = []
        for row in balance_sheet:
            item = row[0].__dict__
            bs_data.extend(item["balance_sheet"])
        
        cf_data = []
        for row in cashflow:
            item = row[0].__dict__
            cf_data.extend(item["cashflow"])
        
        is_data = []
        for row in income_statement:
            item = row[0].__dict__
            is_data.extend(item["income_statement"])
        
        fr_data = []
        for row in financial_ratio:
            item = row[0].__dict__
            fr_data.extend(item["financial_ratio"])
        
        df_balancesheet = pd.DataFrame(bs_data)
        df_cashflow = pd.DataFrame(cf_data)
        df_incomestatement = pd.DataFrame(is_data)
        df_financialratio = pd.DataFrame(fr_data)
        
        # Merge data
        df_merged = df_balancesheet[['year', 'asset', 'debt', 'equity']].merge(
            df_cashflow[['year', 'freeCashFlow']], on='year', how='inner'
        ).merge(
            df_incomestatement[['year', 'revenue', 'postTaxProfit']], on='year', how='inner'
        ).merge(
            df_financialratio[['year', 'roe', 'roa']], on='year', how='inner'
        )
        
        # Create correlation heatmap
        corr_matrix = df_merged[['asset', 'debt', 'equity', 'freeCashFlow', 'revenue', 'postTaxProfit', 'roe', 'roa']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title(f'{symbol} Correlation Matrix of Key Financial Features')
        
        img_base64 = create_base64_image(fig)
        
        return {
            "status": True,
            "image": img_base64,
            "symbol": symbol,
            "data_points": len(df_merged)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating correlation chart: {str(e)}")
