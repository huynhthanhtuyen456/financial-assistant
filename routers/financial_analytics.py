import io
import base64
import math
from typing import List, Optional
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlmodel import select
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    symbol: str = Query("FPT", description="Stock symbol to analyze"),
    yearly: bool = Query(True, description="Use yearly data"),
    session: AsyncSession = Depends(session_manager.session)
):
    """Get financial analytics dashboard HTML page"""
    
    # Fetch data from database
    try:
        # Get balance sheet data
        # stmt_bs = (select(BalanceSheet)
        #           .where(BalanceSheet.symbol == symbol)
        #           .where(BalanceSheet.yearly == yearly))
        # queryset_bs = await session.execute(stmt_bs)
        # balance_sheet = queryset_bs.fetchall()
        balance_sheet = requests.get(f'https://app.finsc.vn/api/v1/scfa/balancesheet?symbols={symbol}&yearly={yearly}')
        balance_sheet = balance_sheet.json()
        df_balancesheet = pd.DataFrame(balance_sheet['data'])
        
        if not balance_sheet:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Extract data
        # bs_data = []
        # for row in balance_sheet:
        #     item = row[0].__dict__
        #     bs_data.extend(item["balance_sheet"])
        
        # df_balancesheet = pd.DataFrame(bs_data)
        
        # Get cashflow data
        # stmt_cf = (select(Cashflow)
        #           .where(Cashflow.symbol == symbol)
        #           .where(Cashflow.yearly == yearly))
        # queryset_cf = await session.execute(stmt_cf)
        # cashflow = queryset_cf.fetchall()
        cashflow = requests.get(f'https://app.finsc.vn/api/v1/scfa/cashflow?symbols={symbol}&yearly={yearly}')
        cashflow = cashflow.json()
        df_cashflow = pd.DataFrame(cashflow['data'])
        
        # cf_data = []
        # for row in cashflow:
        #     item = row[0].__dict__
        #     cf_data.extend(item["cashflow"])
        
        # df_cashflow = pd.DataFrame(cf_data)
        
        # Get income statement data
        # stmt_is = (select(IncomeStatement)
        #           .where(IncomeStatement.symbol == symbol)
        #           .where(IncomeStatement.yearly == yearly))
        # queryset_is = await session.execute(stmt_is)
        # income_statement = queryset_is.fetchall()
        income_statement = requests.get(f'https://app.finsc.vn/api/v1/scfa/incomestatement?symbols={symbol}&yearly={yearly}')
        income_statement = income_statement.json()
        df_incomestatement = pd.DataFrame(income_statement['data'])
        
        # is_data = []
        # for row in income_statement:
        #     item = row[0].__dict__
        #     is_data.extend(item["income_statement"])
        
        # df_incomestatement = pd.DataFrame(is_data)
        
        # Get financial ratio data
        # stmt_fr = (select(FinancialRatio)
        #           .where(FinancialRatio.symbol == symbol)
        #           .where(FinancialRatio.yearly == yearly))
        # queryset_fr = await session.execute(stmt_fr)
        # financial_ratio = queryset_fr.fetchall()
        financial_ratio = requests.get(f'https://app.finsc.vn/api/v1/scfa/financialratio?symbols={symbol}&yearly={yearly}')
        financial_ratio = financial_ratio.json()
        df_financialratio = pd.DataFrame(financial_ratio['data'])
        
        # fr_data = []
        # for row in financial_ratio:
        #     item = row[0].__dict__
        #     fr_data.extend(item["financial_ratio"])
        
        # df_financialratio = pd.DataFrame(fr_data)
        
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
        ax.set_title(f'{symbol} Balancesheet: Assets, Debt, and Equity Over Years')
        ax.legend()
        ax.grid(True)
        visualizations['asset_debt_equity'] = create_base64_image(fig)
        
        # 2. Short and Long Debt trends
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_bs_sorted['year'], df_bs_sorted['longDebt'], label='Long Debt', marker='o')
        ax.plot(df_bs_sorted['year'], df_bs_sorted['shortDebt'], label='Short Debt', marker='s')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value (Billion VND)')
        ax.set_title(f'{symbol} Balancesheet: Short and Long Debt Over Years')
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
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Analytics Dashboard - {symbol}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #007bff;
            }}
            .header h1 {{
                color: #007bff;
                margin: 0;
            }}
            .chart-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .chart-container h2 {{
                color: #333;
                margin-bottom: 20px;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .stats-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                color: #333;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }}
            .form-container {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .form-group {{
                margin-bottom: 15px;
            }}
            .form-group label {{
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }}
            .form-group input, .form-group select {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            .btn {{
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            .btn:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Financial Analytics Dashboard</h1>
                <p>Comprehensive financial analysis for {symbol}</p>
            </div>
            
            <div class="form-container">
                <form method="get">
                    <div class="form-group">
                        <label for="symbol">Stock Symbol:</label>
                        <input type="text" id="symbol" name="symbol" value="{symbol}" required>
                    </div>
                    <div class="form-group">
                        <label for="yearly">Data Type:</label>
                        <select id="yearly" name="yearly">
                            <option value="true" {'selected' if yearly else ''}>Yearly</option>
                            <option value="false" {'selected' if not yearly else ''}>Quarterly</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">Update Analysis</button>
                </form>
            </div>
            
            <div class="stats-container">
                <div class="stat-card">
                    <h3>Total Assets (Latest)</h3>
                    <div class="stat-value">{df_bs_sorted['asset'].iloc[-1]:,.0f} B VND</div>
                </div>
                <div class="stat-card">
                    <h3>Total Debt (Latest)</h3>
                    <div class="stat-value">{df_bs_sorted['debt'].iloc[-1]:,.0f} B VND</div>
                </div>
                <div class="stat-card">
                    <h3>Equity (Latest)</h3>
                    <div class="stat-value">{df_bs_sorted['equity'].iloc[-1]:,.0f} B VND</div>
                </div>
                <div class="stat-card">
                    <h3>Revenue (Latest)</h3>
                    <div class="stat-value">{df_is_sorted['revenue'].iloc[-1]:,.0f} B VND</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Asset, Debt, and Equity Trends</h2>
                <img src="data:image/png;base64,{visualizations.get('asset_debt_equity', '')}" alt="Asset, Debt, Equity Chart">
            </div>
            
            <div class="chart-container">
                <h2>Debt Breakdown (Short vs Long Term)</h2>
                <img src="data:image/png;base64,{visualizations.get('debt_breakdown', '')}" alt="Debt Breakdown Chart">
            </div>
            
            <div class="chart-container">
                <h2>Free Cash Flow Trend</h2>
                <img src="data:image/png;base64,{visualizations.get('free_cashflow', '')}" alt="Free Cash Flow Chart">
            </div>
            
            <div class="chart-container">
                <h2>Revenue and Net Profit Trend</h2>
                <img src="data:image/png;base64,{visualizations.get('revenue_profit', '')}" alt="Revenue and Profit Chart">
            </div>
            
            <div class="chart-container">
                <h2>Financial Metrics Correlation</h2>
                <img src="data:image/png;base64,{visualizations.get('correlation_heatmap', '')}" alt="Correlation Heatmap">
            </div>
            
            {f'''
            <div class="chart-container">
                <h2>Revenue Prediction Model (R² = {r2_score:.3f})</h2>
                <img src="data:image/png;base64,{visualizations.get('revenue_prediction', '')}" alt="Revenue Prediction Chart">
            </div>
            ''' if r2_score is not None else ''}
            
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

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
