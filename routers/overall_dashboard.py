import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.helpers import get_symbols, get_balance_sheet, get_cash_flow, get_income_statement
from db import session_manager
from models.balancesheet import BalanceSheet
from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.financial_ratio import FinancialRatio

router = APIRouter(
    prefix="/dashboard",
    tags=["overall-financial-dashboard"]
)
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def overall_dashboard(
    request: Request,
    session: AsyncSession = Depends(session_manager.session),
    symbol: str = Query('FPT', description="Stock symbol"),
    prediction_year: int = Query(2023, description="Year to predict"),
    yearly: bool = Query(True, description="Use yearly data"),
):
    """
    Overall dashboard combining Balance Sheet, Cash Flow, and Income Statement analysis
    """
    symbol = symbol.upper()
    
    try:
        # Fetch all data sources
        df_balance_sheet = await get_balance_sheet(session, symbol, prediction_year, yearly)
        df_cashflow = await get_cash_flow(session, symbol, prediction_year, yearly)
        df_income_statement = await get_income_statement(session, symbol, prediction_year, yearly)
        
        # Get financial ratio data
        stmt_fr = (select(FinancialRatio)
                  .where(FinancialRatio.yearly == yearly))
        queryset_fr = await session.execute(stmt_fr)
        financial_ratio = queryset_fr.fetchall()
        
        fr_data = []
        for row in financial_ratio:
            item = row[0].__dict__
            fr_data.extend(item["financial_ratio"])
        
        df_financial_ratio = pd.DataFrame(fr_data) if fr_data else pd.DataFrame()
        
        # Get symbols for dropdown
        symbols = await get_symbols(session)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    
    # Process Balance Sheet data
    df_bs = df_balance_sheet[df_balance_sheet['ticker'] == symbol].copy() if 'ticker' in df_balance_sheet.columns else df_balance_sheet.copy()
    df_bs = df_bs.sort_values('year').reset_index(drop=True)
    df_bs = df_bs.fillna(0)
    
    # Process Cash Flow data
    df_cf = df_cashflow[df_cashflow['ticker'] == symbol].copy() if 'ticker' in df_cashflow.columns else df_cashflow.copy()
    df_cf = df_cf.sort_values('year').reset_index(drop=True)
    df_cf = df_cf.fillna(0)
    
    # Merge balance sheet equity with cashflow for FCFE calculation
    if 'equity' in df_bs.columns and 'year' in df_bs.columns:
        df_cf = df_cf.merge(df_bs[['year', 'equity']], on='year', how='left')
        df_cf['equity'] = df_cf['equity'].fillna(0)
        df_cf['fcfe'] = df_cf.get('fromSale', pd.Series(0)) - df_cf.get('investCost', pd.Series(0)) + df_cf['equity']
    
    # Process Income Statement data
    df_is = df_income_statement[df_income_statement['ticker'] == symbol].copy() if 'ticker' in df_income_statement.columns else df_income_statement.copy()
    df_is = df_is.sort_values('year').reset_index(drop=True)
    df_is = df_is.fillna(0)
    
    # Calculate key metrics
    if len(df_is) > 0 and 'revenue' in df_is.columns:
        df_is['gross_profit_margin'] = ((df_is.get('grossProfit', 0) / (df_is['revenue'] + 1)) * 100)
        df_is['operating_profit_margin'] = ((df_is.get('preTaxProfit', 0) / (df_is['revenue'] + 1)) * 100)
        df_is['net_profit_margin'] = ((df_is.get('postTaxProfit', 0) / (df_is['revenue'] + 1)) * 100)
    
    # Process Financial Ratio data
    df_fr = df_financial_ratio[df_financial_ratio['ticker'] == symbol].copy() if len(df_financial_ratio) > 0 and 'ticker' in df_financial_ratio.columns else pd.DataFrame()
    df_fr = df_fr.sort_values('year').reset_index(drop=True) if len(df_fr) > 0 else pd.DataFrame()
    
    # Merge all data by year for comprehensive analysis
    merged_data = pd.DataFrame()
    if len(df_bs) > 0:
        merged_data = df_bs[['year']].copy()
        if 'asset' in df_bs.columns:
            merged_data = merged_data.merge(df_bs[['year', 'asset', 'debt', 'equity']], on='year', how='left')
    if len(df_cf) > 0:
        if len(merged_data) > 0:
            merged_data = merged_data.merge(df_cf[['year', 'freeCashFlow', 'fcfe']], on='year', how='outer')
        else:
            merged_data = df_cf[['year', 'freeCashFlow', 'fcfe']].copy()
    if len(df_is) > 0:
        if len(merged_data) > 0:
            merged_data = merged_data.merge(df_is[['year', 'revenue', 'postTaxProfit', 'gross_profit_margin', 'net_profit_margin']], on='year', how='outer')
        else:
            merged_data = df_is[['year', 'revenue', 'postTaxProfit', 'gross_profit_margin', 'net_profit_margin']].copy()
    if len(df_fr) > 0 and 'year' in df_fr.columns:
        # Only merge columns that exist in df_fr
        fr_columns = ['year']
        available_ratio_cols = ['roe', 'roa', 'currentRatio', 'quickRatio', 'debtRatio']
        for col in available_ratio_cols:
            if col in df_fr.columns:
                fr_columns.append(col)
        
        if len(merged_data) > 0 and len(fr_columns) > 1:
            merged_data = merged_data.merge(df_fr[fr_columns], on='year', how='outer')
        elif len(fr_columns) > 1:
            merged_data = df_fr[fr_columns].copy()
    
    merged_data = merged_data.sort_values('year').reset_index(drop=True)
    merged_data = merged_data.fillna(0)
    
    # Calculate summary statistics
    summary = {}
    
    # Latest values
    if len(df_bs) > 0:
        summary['latest_asset'] = float(df_bs['asset'].iloc[-1]) if 'asset' in df_bs.columns else 0
        summary['latest_debt'] = float(df_bs['debt'].iloc[-1]) if 'debt' in df_bs.columns else 0
        summary['latest_equity'] = float(df_bs['equity'].iloc[-1]) if 'equity' in df_bs.columns else 0
    
    if len(df_cf) > 0:
        summary['latest_fcf'] = float(df_cf['freeCashFlow'].iloc[-1]) if 'freeCashFlow' in df_cf.columns else 0
        summary['latest_fcfe'] = float(df_cf['fcfe'].iloc[-1]) if 'fcfe' in df_cf.columns else 0
    
    if len(df_is) > 0:
        summary['latest_revenue'] = float(df_is['revenue'].iloc[-1]) if 'revenue' in df_is.columns else 0
        summary['latest_profit'] = float(df_is['postTaxProfit'].iloc[-1]) if 'postTaxProfit' in df_is.columns else 0
        summary['latest_profit_margin'] = float(df_is['net_profit_margin'].iloc[-1]) if 'net_profit_margin' in df_is.columns else 0
    
    # Growth rates
    if len(df_bs) > 1 and 'asset' in df_bs.columns:
        summary['asset_growth'] = ((df_bs['asset'].iloc[-1] / df_bs['asset'].iloc[0]) - 1) * 100 if df_bs['asset'].iloc[0] != 0 else 0
    if len(df_is) > 1 and 'revenue' in df_is.columns:
        summary['revenue_growth'] = ((df_is['revenue'].iloc[-1] / df_is['revenue'].iloc[0]) - 1) * 100 if df_is['revenue'].iloc[0] != 0 else 0
    if len(df_is) > 1 and 'postTaxProfit' in df_is.columns:
        summary['profit_growth'] = ((df_is['postTaxProfit'].iloc[-1] / df_is['postTaxProfit'].iloc[0]) - 1) * 100 if df_is['postTaxProfit'].iloc[0] != 0 else 0
    
    # Financial ratios (latest)
    if len(df_fr) > 0:
        summary['latest_roe'] = float(df_fr['roe'].iloc[-1]) if 'roe' in df_fr.columns else 0
        summary['latest_roa'] = float(df_fr['roa'].iloc[-1]) if 'roa' in df_fr.columns else 0
        summary['latest_current_ratio'] = float(df_fr['currentRatio'].iloc[-1]) if 'currentRatio' in df_fr.columns else 0
        summary['latest_debt_ratio'] = float(df_fr['debtRatio'].iloc[-1]) if 'debtRatio' in df_fr.columns else 0
    
    # Create visualizations using Plotly
    visualizations = {}
    
    # 1. Combined Financial Overview Chart
    if len(merged_data) > 0 and 'year' in merged_data.columns:
        fig_overview = go.Figure()
        
        if 'asset' in merged_data.columns:
            fig_overview.add_trace(go.Scatter(
                x=merged_data['year'],
                y=merged_data['asset'],
                mode='lines+markers',
                name='Total Assets',
                line=dict(color='#1f77b4', width=2)
            ))
        if 'revenue' in merged_data.columns:
            fig_overview.add_trace(go.Scatter(
                x=merged_data['year'],
                y=merged_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#2ca02c', width=2)
            ))
        if 'freeCashFlow' in merged_data.columns:
            fig_overview.add_trace(go.Scatter(
                x=merged_data['year'],
                y=merged_data['freeCashFlow'],
                mode='lines+markers',
                name='Free Cash Flow',
                line=dict(color='#ff7f0e', width=2)
            ))
        
        fig_overview.update_layout(
            title=f'{symbol} - Combined Financial Overview',
            xaxis_title='Year',
            yaxis_title='Value (Billion VND)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        visualizations['overview'] = fig_overview.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 2. Profitability Trends
    if len(df_is) > 0 and 'year' in df_is.columns:
        fig_profitability = go.Figure()
        
        if 'revenue' in df_is.columns:
            fig_profitability.add_trace(go.Scatter(
                x=df_is['year'],
                y=df_is['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='blue', width=2),
                yaxis='y'
            ))
        if 'postTaxProfit' in df_is.columns:
            fig_profitability.add_trace(go.Scatter(
                x=df_is['year'],
                y=df_is['postTaxProfit'],
                mode='lines+markers',
                name='Net Profit',
                line=dict(color='green', width=2),
                yaxis='y'
            ))
        if 'net_profit_margin' in df_is.columns:
            fig_profitability.add_trace(go.Scatter(
                x=df_is['year'],
                y=df_is['net_profit_margin'],
                mode='lines+markers',
                name='Net Profit Margin (%)',
                line=dict(color='red', width=2, dash='dash'),
                yaxis='y2'
            ))
        
        fig_profitability.update_layout(
            title=f'{symbol} - Profitability Trends',
            xaxis_title='Year',
            yaxis=dict(title='Value (Billion VND)', side='left'),
            yaxis2=dict(title='Margin (%)', side='right', overlaying='y'),
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        visualizations['profitability'] = fig_profitability.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 3. Cash Flow Analysis
    if len(df_cf) > 0 and 'year' in df_cf.columns:
        fig_cashflow = go.Figure()
        
        if 'freeCashFlow' in df_cf.columns:
            fig_cashflow.add_trace(go.Scatter(
                x=df_cf['year'],
                y=df_cf['freeCashFlow'],
                mode='lines+markers',
                name='Free Cash Flow',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ))
        if 'fcfe' in df_cf.columns:
            fig_cashflow.add_trace(go.Scatter(
                x=df_cf['year'],
                y=df_cf['fcfe'],
                mode='lines+markers',
                name='FCFE (Free Cash Flow to Equity)',
                line=dict(color='blue', width=2, dash='dash')
            ))
        
        fig_cashflow.update_layout(
            title=f'{symbol} - Cash Flow Analysis',
            xaxis_title='Year',
            yaxis_title='Value (Billion VND)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        visualizations['cashflow'] = fig_cashflow.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 4. Balance Sheet Structure
    if len(df_bs) > 0 and 'year' in df_bs.columns:
        fig_balancesheet = go.Figure()
        
        if 'asset' in df_bs.columns:
            fig_balancesheet.add_trace(go.Scatter(
                x=df_bs['year'],
                y=df_bs['asset'],
                mode='lines+markers',
                name='Total Assets',
                line=dict(color='#1f77b4', width=2)
            ))
        if 'debt' in df_bs.columns:
            fig_balancesheet.add_trace(go.Scatter(
                x=df_bs['year'],
                y=df_bs['debt'],
                mode='lines+markers',
                name='Total Debt',
                line=dict(color='#d62728', width=2)
            ))
        if 'equity' in df_bs.columns:
            fig_balancesheet.add_trace(go.Scatter(
                x=df_bs['year'],
                y=df_bs['equity'],
                mode='lines+markers',
                name='Equity',
                line=dict(color='#2ca02c', width=2)
            ))
        
        fig_balancesheet.update_layout(
            title=f'{symbol} - Balance Sheet Structure',
            xaxis_title='Year',
            yaxis_title='Value (Billion VND)',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        visualizations['balancesheet'] = fig_balancesheet.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 5. Financial Ratios
    if len(df_fr) > 0 and 'year' in df_fr.columns:
        fig_ratios = go.Figure()
        
        if 'roe' in df_fr.columns:
            fig_ratios.add_trace(go.Scatter(
                x=df_fr['year'],
                y=df_fr['roe'],
                mode='lines+markers',
                name='ROE (%)',
                line=dict(color='blue', width=2)
            ))
        if 'roa' in df_fr.columns:
            fig_ratios.add_trace(go.Scatter(
                x=df_fr['year'],
                y=df_fr['roa'],
                mode='lines+markers',
                name='ROA (%)',
                line=dict(color='green', width=2)
            ))
        if 'currentRatio' in df_fr.columns and len(df_fr) > 0:
            fig_ratios.add_trace(go.Scatter(
                x=df_fr['year'],
                y=df_fr['currentRatio'],
                mode='lines+markers',
                name='Current Ratio',
                line=dict(color='orange', width=2, dash='dash')
            ))
        
        fig_ratios.update_layout(
            title=f'{symbol} - Financial Ratios',
            xaxis_title='Year',
            yaxis_title='Ratio Value',
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )
        visualizations['ratios'] = fig_ratios.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Prepare context
    context = {
        "request": request,
        "symbol": symbol,
        "symbols": symbols,
        "prediction_year": prediction_year,
        "yearly": yearly,
        "summary": summary,
        "visualizations": visualizations,
        "df_bs": df_bs.to_dict('records') if len(df_bs) > 0 else [],
        "df_cf": df_cf.to_dict('records') if len(df_cf) > 0 else [],
        "df_is": df_is.to_dict('records') if len(df_is) > 0 else [],
        "df_fr": df_fr.to_dict('records') if len(df_fr) > 0 else [],
        "merged_data": merged_data.to_dict('records') if len(merged_data) > 0 else [],
    }
    
    return templates.TemplateResponse("overall_dashboard.html", context=context)

