import math
import plotly.graph_objects as go

from fastapi.templating import Jinja2Templates

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from db import session_manager
import pandas as pd
import numpy as np
import plotly.express as px
from core.helpers import get_balance_sheet, get_income_statement, get_cash_flow
from plotly.subplots import make_subplots


router = APIRouter(
    prefix="/eda",
    tags=["eda-dashboard"]
)
templates = Jinja2Templates(directory="templates")


@router.get("/balance-sheet", response_class=HTMLResponse)
async def eda_balance_sheet(request: Request, session: AsyncSession = Depends(session_manager.session)):
    """Prepairing Data"""
    df_bs = await get_balance_sheet(session, yearly=True)
    numeric_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
    df_bs.sort_values(by=['ticker', 'year'], inplace=True)
    df_bs.reset_index(drop=True, inplace=True)
    df_bs[numeric_cols].fillna(0, inplace=True)
    # Melt the dataframe to long format for easier plotting
    melted = df_bs[numeric_cols].melt(var_name='variable', value_name='value')
    df_bs["currentLiabilities"] = df_bs["payable"] + df_bs["shortDebt"] + df_bs["longDebt"]
    df_bs["currentAsset"] = df_bs["cash"] + df_bs["shortReceivable"] + \
                                       df_bs["inventory"] + df_bs["shortAsset"] + \
                                       df_bs["otherDebt"]
    df_bs["currentRatio"] = df_bs["currentAsset"] / df_bs["currentLiabilities"]
    df_bs["currentRatio"] = df_bs["currentRatio"].replace([np.inf, -np.inf], 0)

    """
    Correlation Metrix
    """
    # Get balance sheet data for correlation analysis
    # Select multiple features for correlation
    # correlation_features = ['asset', 'debt', 'equity', 'cash', 'payable']

    # Filter data to include only rows with complete feature data
    df_corr = df_bs[numeric_cols].copy()

    # Calculate correlation matrix
    correlation_matrix = df_corr.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values = []
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if not pd.isna(corr_value):
                corr_values.append(corr_value)

    mean_corr = np.mean(corr_values) if corr_values else 0
    median_corr = np.median(corr_values) if corr_values else 0
    std_corr = np.std(corr_values) if corr_values else 0

    # Create interactive heatmap using Plotly
    corr_fig = go.Figure()
    fig = corr_fig.add_trace(go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Balance Sheet Features<br><sub>Mean: '
                    f'{mean_corr:.3f}, Median: {median_corr:.3f}, Std: {std_corr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    fig_corr_metrix_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Correlation Metrix
    """

    """Create distribution plots for each numeric feature"""
    # Create distribution plots for each numeric feature
    fig = px.box(melted, x='variable', y='value',
                 title='Distribution of Numeric Features in Balance Sheet',
                 labels={'variable': 'Feature', 'value': 'Value'},
                 height=600, width=1200)

    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        hovermode='closest'
    )

    fig.show()

    # Alternative: Violin plot for better distribution visualization
    fig_violin = px.violin(melted, x='variable', y='value',
                           title='Distribution of Numeric Features (Violin Plot)',
                           labels={'variable': 'Feature', 'value': 'Value'},
                           height=600, width=800,
                           points='outliers')

    fig_violin.update_layout(xaxis_tickangle=-45)
    fig_violin_html = fig_violin.to_html(full_html=False, include_plotlyjs='cdn')

    # Time series trend: Average value per year for each feature
    trend_by_year = df_bs[numeric_cols].groupby(df_bs['year']).mean()
    trend_melted = trend_by_year.melt(id_vars='year', var_name='feature', value_name='avg_value')

    fig_trend = px.line(trend_melted, x='year', y='avg_value', color='feature',
                        title='Trend of Features Over Time (Average per Year)',
                        labels={'year': 'Year', 'avg_value': 'Average Value', 'feature': 'Feature'},
                        height=600, width=800)

    fig_trend_html = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')
    """Create distribution plots for each numeric feature"""

    """
    Asset Trend
    """
    # Plot asset history for all companies and overall yearly average
    fig_asset = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_bs['ticker'].unique():
        df_ticker = df_bs[df_bs['ticker'] == ticker].sort_values('year')
        fig_asset.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['asset'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_year_avg = df_bs.groupby(['year'], as_index=False)['asset'].mean()
    fig_asset.add_trace(go.Scatter(
        x=df_year_avg['year'],
        y=df_year_avg['asset'],
        mode='lines',
        name='Yearly Average (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_asset.update_layout(
        title='Total Asset Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Asset',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_asset_trend_html = fig_asset.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Asset Trend
    """

    """
    Debt Trend
    """
    # Plot asset history for all companies and overall yearly average
    fig_debt = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_bs['ticker'].unique():
        df_ticker = df_bs[df_bs['ticker'] == ticker].sort_values('year')
        fig_debt.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['debt'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_year_debt_avg = df_bs.groupby(['year'], as_index=False)['debt'].mean()
    fig_debt.add_trace(go.Scatter(
        x=df_year_debt_avg['year'],
        y=df_year_debt_avg['debt'],
        mode='lines',
        name='Yearly Average (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_debt.update_layout(
        title='Total Debt Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Debt',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_debt_trend_html = fig_debt.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Debt Trend
    """

    """
    Equity Trend
    """
    # Plot asset history for all companies and overall yearly average
    fig_equity = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_bs['ticker'].unique():
        df_ticker = df_bs[df_bs['ticker'] == ticker].sort_values('year')
        fig_equity.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['equity'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_year_equity_avg = df_bs.groupby(['year'], as_index=False)['equity'].mean()
    fig_equity.add_trace(go.Scatter(
        x=df_year_equity_avg['year'],
        y=df_year_equity_avg['equity'],
        mode='lines',
        name='Yearly Average (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_equity.update_layout(
        title='Total Equity Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Debt',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_equity_trend_html = fig_equity.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Equity Trend
    """

    """
    Average Current Ratio
    """
    # Calculate average currentRatio by symbol (across all years)
    avg_cr_by_symbol = df_bs.groupby('ticker')['currentRatio'].mean().reset_index()
    avg_cr_by_symbol.columns = ['ticker', 'avg_currentRatio']
    avg_cr_by_symbol = avg_cr_by_symbol.sort_values('avg_currentRatio', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_cr = avg_cr_by_symbol.head(50)
    top_50_lowest_cr = avg_cr_by_symbol.tail(50)

    # Create subplots: 1 row, 2 columns
    top_50_highest_cr_mean = top_50_highest_cr['avg_currentRatio'].mean()
    top_50_highest_cr_md = top_50_highest_cr['avg_currentRatio'].median()
    top_50_highest_cr_std = top_50_highest_cr['avg_currentRatio'].std()

    top_50_lowest_cr_mean = top_50_lowest_cr['avg_currentRatio'].mean()
    top_50_lowest_cr_md = top_50_lowest_cr['avg_currentRatio'].median()
    top_50_lowest_cr_std = top_50_lowest_cr['avg_currentRatio'].std()

    titles = [
        f'Top 50 Highest Average Current Ratio - Mean: '
        f'{top_50_highest_cr_mean:.3f}, Median: {top_50_highest_cr_md:.3f}, Std: {top_50_highest_cr_std:.3f}<b>',
        f'Top 50 Lowest Average Current Ratio - Mean: '
        f'{top_50_lowest_cr_mean:.3f}, Median: {top_50_lowest_cr_md:.3f}, Std: {top_50_lowest_cr_std:.3f}<b>',
    ]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig.add_trace(
        go.Bar(
            x=top_50_highest_cr['avg_currentRatio'],
            y=top_50_highest_cr['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Current Ratio: %{x:.2f}<extra></extra>',
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig.add_trace(
        go.Bar(
            x=top_50_lowest_cr['avg_currentRatio'],
            y=top_50_lowest_cr['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Current Ratio: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Distribution of Average Current Ratio by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Average Current Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Average Current Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Symbol", row=1, col=1)
    fig.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_cr_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Average Current Ratio
    """

    """Start Quick Ratio"""
    df_bs["quickRatio"] = np.where(
        df_bs["shortDebt"] != 0,
        (df_bs["shortAsset"] - df_bs["inventory"]) / df_bs["shortDebt"],
        df_bs["shortAsset"] - df_bs["inventory"]
    )

    # Calculate average quickRatio by symbol (across all years)
    avg_qr_by_symbol = df_bs.groupby('ticker')['quickRatio'].mean().reset_index()
    avg_qr_by_symbol.columns = ['ticker', 'avg_quickRatio']
    avg_qr_by_symbol = avg_qr_by_symbol.sort_values('avg_quickRatio', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_qr = avg_qr_by_symbol.head(50)
    top_50_lowest_qr = avg_qr_by_symbol.tail(50)

    top_50_highest_qr_mean = top_50_highest_qr['avg_quickRatio'].mean()
    top_50_highest_qr_md = top_50_highest_qr['avg_quickRatio'].median()
    top_50_highest_qr_std = top_50_highest_qr['avg_quickRatio'].std()

    top_50_lowest_qr_mean = top_50_lowest_qr['avg_quickRatio'].mean()
    top_50_lowest_qr_md = top_50_lowest_qr['avg_quickRatio'].median()
    top_50_lowest_qr_std = top_50_lowest_qr['avg_quickRatio'].std()

    qr_titles = [
        f'Top 50 Highest Average Quick Ratio - Mean: '
        f'{top_50_highest_qr_mean:.3f}, Median: {top_50_highest_qr_md:.3f}, Std: {top_50_highest_qr_std:.3f}<b>',
        f'Top 50 Lowest Average Quick Ratio - Mean: '
        f'{top_50_lowest_qr_mean:.3f}, Median: {top_50_lowest_qr_md:.3f}, Std: {top_50_lowest_qr_std:.3f}<b>',
    ]

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=qr_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig.add_trace(
        go.Bar(
            x=top_50_highest_qr['avg_quickRatio'],
            y=top_50_highest_qr['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Quick Ratio: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig.add_trace(
        go.Bar(
            x=top_50_lowest_qr['avg_quickRatio'],
            y=top_50_lowest_qr['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Quick Ratio: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Distribution of Average Quick Ratio by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Average Quick Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Average Quick Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Symbol", row=1, col=1)
    fig.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_qr_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """End Quick Ratio"""

    """Start Debt to Equity Ratio"""
    df_bs["debtToEquity"] = np.where(
        df_bs["equity"] != 0,
        (df_bs["debt"] + df_bs["otherDebt"]) / df_bs["equity"],
        df_bs["debt"] + df_bs["otherDebt"]
    )

    # Calculate average quickRatio by symbol (across all years)
    avg_dbe_by_symbol = df_bs.groupby('ticker')['debtToEquity'].mean().reset_index()
    avg_dbe_by_symbol.columns = ['ticker', 'avg_debtToEquity']
    avg_dbe_by_symbol = avg_dbe_by_symbol.sort_values('avg_debtToEquity', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_dbe = avg_dbe_by_symbol.head(50)
    top_50_lowest_dbe = avg_dbe_by_symbol.tail(50)

    top_50_highest_dbe_mean = top_50_highest_dbe['avg_debtToEquity'].mean()
    top_50_highest_dbe_md = top_50_highest_dbe['avg_debtToEquity'].median()
    top_50_highest_dbe_std = top_50_highest_dbe['avg_debtToEquity'].std()

    top_50_lowest_dbe_mean = top_50_lowest_dbe['avg_debtToEquity'].mean()
    top_50_lowest_dbe_md = top_50_lowest_dbe['avg_debtToEquity'].median()
    top_50_lowest_dbe_std = top_50_lowest_dbe['avg_debtToEquity'].std()

    dbe_titles = [
        f'Top 50 Highest Average Debt To Equity Ratio - Mean: '
        f'{top_50_highest_dbe_mean:.3f}, Median: {top_50_highest_dbe_md:.3f}, Std: {top_50_highest_dbe_std:.3f}<b>',
        f'Top 50 Lowest Average Debt To Equity - Mean: '
        f'{top_50_lowest_dbe_mean:.3f}, Median: {top_50_lowest_dbe_md:.3f}, Std: {top_50_lowest_dbe_std:.3f}<b>',
    ]

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=dbe_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig.add_trace(
        go.Bar(
            x=top_50_highest_dbe['avg_debtToEquity'],
            y=top_50_highest_dbe['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Debt To Equity Ratio: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig.add_trace(
        go.Bar(
            x=top_50_lowest_dbe['avg_debtToEquity'],
            y=top_50_lowest_dbe['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Debt To Equity Ratio: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Distribution of Average Debt To Equity Ratio by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(title_text="Average Debt To Equity Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Average Debt To Equity Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Symbol", row=1, col=1)
    fig.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_dbe_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """End Debt to Equity Ratio"""

    """
    Financial Ratio Correlation Metrix
    """
    features_cr = ['currentLiabilities', 'currentRatio', 'currentAsset', 'payable', 'shortDebt', 'longDebt', 'cash',
                   'shortReceivable', 'inventory', 'shortAsset', 'otherDebt']
    dbe_features = ["debt", "otherDebt", "equity", 'debtToEquity']
    qr_features = ["shortAsset", "inventory", "shortDebt", 'quickRatio']

    # Filter data to include only rows with complete feature data
    df_corr_cr = df_bs[features_cr].copy()

    # Calculate correlation matrix
    correlation_matrix_cr = df_corr_cr.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_cr = []
    for i in range(len(correlation_matrix_cr)):
        for j in range(i + 1, len(correlation_matrix_cr)):
            corr_value_cr = correlation_matrix_cr.iloc[i, j]
            if not pd.isna(corr_value_cr):
                corr_values_cr.append(corr_value_cr)

    mean_corr_cr = np.mean(corr_values_cr) if corr_values_cr else 0
    median_corr_cr = np.median(corr_values_cr) if corr_values_cr else 0
    std_corr_cr = np.std(corr_values_cr) if corr_values_cr else 0

    # Create interactive heatmap using Plotly
    corr_fig_cr = go.Figure()
    corr_fig_cr = corr_fig_cr.add_trace(go.Heatmap(
        z=correlation_matrix_cr.values,
        x=correlation_matrix_cr.columns,
        y=correlation_matrix_cr.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_cr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_cr.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Current Ratio Indicator Features<br><sub>Mean: '
                    f'{mean_corr_cr:.3f}, Median: {median_corr_cr:.3f}, Std: {std_corr_cr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_cr_metrix_html = corr_fig_cr.to_html(full_html=False, include_plotlyjs='cdn')

    # Fig Corr QR
    # Filter data to include only rows with complete feature data
    df_corr_qr = df_bs[qr_features].copy()

    # Calculate correlation matrix
    correlation_matrix_qr = df_corr_qr.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_qr = []
    for i in range(len(correlation_matrix_qr)):
        for j in range(i + 1, len(correlation_matrix_qr)):
            corr_value_qr = correlation_matrix_qr.iloc[i, j]
            if not pd.isna(corr_value_qr):
                corr_values_qr.append(corr_value_qr)

    mean_corr_qr = np.mean(corr_values_qr) if corr_values_qr else 0
    median_corr_qr = np.median(corr_values_qr) if corr_values_qr else 0
    std_corr_qr = np.std(corr_values_qr) if corr_values_qr else 0

    # Create interactive heatmap using Plotly
    corr_fig_qr = go.Figure()
    corr_fig_qr.add_trace(go.Heatmap(
        z=correlation_matrix_qr.values,
        x=correlation_matrix_qr.columns,
        y=correlation_matrix_qr.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_qr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_qr.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Quick Ratio Indicator Features<br><sub>Mean: '
                    f'{mean_corr_qr:.3f}, Median: {median_corr_qr:.3f}, Std: {std_corr_qr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_qr_metrix_html = corr_fig_qr.to_html(full_html=False, include_plotlyjs='cdn')

    # Fig Corr DBER
    # Filter data to include only rows with complete feature data
    df_corr_dbe = df_bs[dbe_features].copy()

    # Calculate correlation matrix
    correlation_matrix_dbe = df_corr_dbe.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_dbe = []
    for i in range(len(correlation_matrix_dbe)):
        for j in range(i + 1, len(correlation_matrix_dbe)):
            corr_value_dbe = correlation_matrix_dbe.iloc[i, j]
            if not pd.isna(corr_value_dbe):
                corr_values_dbe.append(corr_value_dbe)

    mean_corr_dbe = np.mean(corr_values_dbe) if corr_values_dbe else 0
    median_corr_dbe = np.median(corr_values_dbe) if corr_values_dbe else 0
    std_corr_dbe = np.std(corr_values_dbe) if corr_values_dbe else 0

    # Create interactive heatmap using Plotly
    corr_fig_dbe = go.Figure()
    corr_fig_dbe.add_trace(go.Heatmap(
        z=correlation_matrix_dbe.values,
        x=correlation_matrix_dbe.columns,
        y=correlation_matrix_dbe.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_dbe.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_dbe.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Debt To Equity Indicator Features<br><sub>Mean: '
                    f'{mean_corr_dbe:.3f}, Median: {median_corr_dbe:.3f}, Std: {std_corr_dbe:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_dbe_metrix_html = corr_fig_dbe.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Financial Ratio Correlation Metrix
    """

    # context var
    context = {
        "request": request,
        "fig_trend_html": fig_trend_html,
        "fig_violin_html": fig_violin_html,
        "fig_corr_metrix_html": fig_corr_metrix_html,
        "corr_fig_cr_metrix_html": corr_fig_cr_metrix_html,
        "corr_fig_qr_metrix_html": corr_fig_qr_metrix_html,
        "corr_fig_dbe_metrix_html": corr_fig_dbe_metrix_html,
        "fig_asset_trend_html": fig_asset_trend_html,
        "fig_debt_trend_html": fig_debt_trend_html,
        "fig_equity_trend_html": fig_equity_trend_html,
        "fig_cr_html": fig_cr_html,
        "fig_qr_html": fig_qr_html,
        "fig_dbe_html": fig_dbe_html,
    }

    return templates.TemplateResponse("eda/balance_sheet.html", context=context)


@router.get("/distribution", response_class=HTMLResponse)
async def distribution(request: Request, session: AsyncSession = Depends(session_manager.session)):
    df_is = await get_income_statement(session, year=2021, symbol="FPT", yearly=True)

    # 1. Prepare the data (Same as your original code)
    income_statement_numerics_cols = df_is.columns.tolist()[3:]  # Exclude 'symbol', 'year', 'quarter'
    df_income_statement_sorted = df_is.sort_values('year')

    # 2. Dynamically determine grid size
    num_plots = len(income_statement_numerics_cols)
    ncols = 4
    nrows = math.ceil(num_plots / ncols)

    # 3. Create the Subplot Figure
    # vertical_spacing creates a bit of breathing room between rows
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=income_statement_numerics_cols,  # Adds titles to each chart automatically
        vertical_spacing=0.05
    )

    # 4. Loop and add traces
    for i, col in enumerate(income_statement_numerics_cols):
        # Calculate Plotly's 1-based row and column indices
        row = (i // ncols) + 1
        c = (i % ncols) + 1
        # --- DYNAMIC UNIT LOGIC ---
        # Check if column name suggests it's a percentage
        is_percent = any(k in col.lower() for k in income_statement_numerics_cols)

        # If it's a percentage, use ".1%" (e.g., 0.15 -> 15.0%)
        # If not, use ".2s" (SI prefixes: 1000 -> 1k, 1000000 -> 1M, 1.5 -> 1.5)
        y_format = ".1%" if is_percent else ".2s"
        # ---------------------------

        fig.add_trace(
            go.Scatter(
                x=df_income_statement_sorted['year'],
                y=df_income_statement_sorted[col],
                mode='lines',  # Equivalent to marker='o' in matplotlib
                name=col,
                hovertemplate=f"Year: %{{x}}<br>{col}: %{{y:{y_format}}}"
            ),
            row=row,
            col=c
        )

    # 5. Update Layout
    fig.update_layout(
        height=300 * nrows,  # Dynamic height: 300px per row
        width=1600,  # Fixed width or adjust as needed
        title_text="Yearly Distribution of Income Statement Metrics",
        title_font_size=18,
        showlegend=False  # Hides the legend to avoid cluttering the view
    )

    """
    Start Distribution
    """
    df_bs = await get_balance_sheet(session, yearly=True)
    # 1. Prepare the data (Same as your original code)
    df_bs_sorted = df_bs.sort_values('year')
    # indicator_cols = ["asset", "debt", "cash", "equity", "payable"]
    indicator_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
    indicator_cols.remove('quarter')
    indicator_cols.remove('year')
    # 2. Dynamically determine grid size
    num_plots = len(indicator_cols)
    ncols = 3
    nrows = math.ceil(num_plots / ncols)

    # 3. Create the Subplot Figure
    # vertical_spacing creates a bit of breathing room between rows
    fig_distribution_bs = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=indicator_cols,  # Adds titles to each chart automatically
        vertical_spacing=0.05
    )

    # 4. Loop and add traces
    for i, col in enumerate(indicator_cols):
        # Calculate Plotly's 1-based row and column indices
        row = (i // ncols) + 1
        c = (i % ncols) + 1
        # --- DYNAMIC UNIT LOGIC ---
        # Check if column name suggests it's a percentage
        is_percent = any(k in col.lower() for k in indicator_cols)

        # If it's a percentage, use ".1%" (e.g., 0.15 -> 15.0%)
        # If not, use ".2s" (SI prefixes: 1000 -> 1k, 1000000 -> 1M, 1.5 -> 1.5)
        y_format = ".1%" if is_percent else ".2s"
        # ---------------------------

        fig_distribution_bs.add_trace(
            go.Scatter(
                x=df_bs_sorted['year'],
                y=df_bs_sorted[col],
                mode='lines',  # Equivalent to marker='o' in matplotlib
                name=col,
                hovertemplate=f"Year: %{{x}}<br>{col}: %{{y:{y_format}}}"
            ),
            row=row,
            col=c
        )

    # 5. Update Layout
    fig_distribution_bs.update_layout(
        height=300 * nrows,  # Dynamic height: 300px per row
        width=1600,  # Fixed width or adjust as needed
        title_text="Yearly Distribution of Balance Sheet Metrics",
        title_font_size=18,
        showlegend=False  # Hides the legend to avoid cluttering the view
    )

    fig_distribution_bs_html = fig_distribution_bs.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Distribution
    """

    """
        Start Distribution
        """
    df_cf = await get_cash_flow(session, yearly=True, symbol="FPT", year=2025)
    # 1. Prepare the data (Same as your original code)
    df_cf_sorted = df_cf.sort_values('year')
    numeric_cols_cf = df_cf.select_dtypes(include=[np.number]).columns.tolist()
    # 2. Dynamically determine grid size
    num_plots = len(numeric_cols_cf)
    ncols = 3
    nrows = math.ceil(num_plots / ncols)

    # 3. Create the Subplot Figure
    # vertical_spacing creates a bit of breathing room between rows
    fig_distribution_cf = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=numeric_cols_cf,  # Adds titles to each chart automatically
        vertical_spacing=0.2
    )

    # 4. Loop and add traces
    for i, col in enumerate(numeric_cols_cf):
        # Calculate Plotly's 1-based row and column indices
        row = (i // ncols) + 1
        c = (i % ncols) + 1
        # --- DYNAMIC UNIT LOGIC ---
        # Check if column name suggests it's a percentage
        is_percent = any(k in col.lower() for k in numeric_cols_cf)

        # If it's a percentage, use ".1%" (e.g., 0.15 -> 15.0%)
        # If not, use ".2s" (SI prefixes: 1000 -> 1k, 1000000 -> 1M, 1.5 -> 1.5)
        y_format = ".1%" if is_percent else ".2s"
        # ---------------------------

        fig_distribution_cf.add_trace(
            go.Scatter(
                x=df_cf_sorted['year'],
                y=df_cf_sorted[col],
                mode='lines',  # Equivalent to marker='o' in matplotlib
                name=col,
                hovertemplate=f"Year: %{{x}}<br>{col}: %{{y:{y_format}}}"
            ),
            row=row,
            col=c
        )

    # 5. Update Layout
    fig_distribution_cf.update_layout(
        height=300 * nrows,  # Dynamic height: 300px per row
        width=1600,  # Fixed width or adjust as needed
        title_text="Yearly Distribution of Cash Flow Metrics",
        title_font_size=18,
        showlegend=False  # Hides the legend to avoid cluttering the view
    )

    fig_distribution_cf_html = fig_distribution_cf.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Distribution
    """

    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    context = {
        "request": request,
        "fig_html": fig_html,
        "fig_distribution_bs_html": fig_distribution_bs_html,
        "fig_distribution_cf_html": fig_distribution_cf_html,
    }
    return templates.TemplateResponse("eda/distribution.html", context=context)


@router.get("/income-statement", response_class=HTMLResponse)
async def eda_income_statement(request: Request, session: AsyncSession = Depends(session_manager.session)):
    df_is = await get_income_statement(session, yearly=True, symbol="FPT", year=2025)
    numeric_cols = df_is.select_dtypes(include=[np.number]).columns.tolist()
    df_is.sort_values(by=['ticker', 'year'], inplace=True)
    df_is.reset_index(drop=True, inplace=True)
    df_is[numeric_cols].fillna(0, inplace=True)

    """
    Correlation Metrix
    """
    # Get balance sheet data for correlation analysis
    # Select multiple features for correlation
    # correlation_features = ['asset', 'debt', 'equity', 'cash', 'payable']

    # Filter data to include only rows with complete feature data
    df_is_corr = df_is[numeric_cols].copy()

    # Calculate correlation matrix
    correlation_matrix = df_is_corr.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values = []
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if not pd.isna(corr_value):
                corr_values.append(corr_value)

    mean_corr = np.mean(corr_values) if corr_values else 0
    median_corr = np.median(corr_values) if corr_values else 0
    std_corr = np.std(corr_values) if corr_values else 0

    # Create interactive heatmap using Plotly
    corr_fig = go.Figure()
    fig = corr_fig.add_trace(go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Income Statement Features<br><sub>Mean: '
                    f'{mean_corr:.3f}, Median: {median_corr:.3f}, Std: {std_corr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    fig_corr_metrix_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Correlation Metrix
    """

    df_is["gross_profit_margin"] = ((df_is["grossProfit"] / df_is["revenue"]) * 100)
    df_is["gross_profit_margin"] = df_is["gross_profit_margin"].replace([np.inf, -np.inf], 0).fillna(0)

    df_is["operating_profit_margin"] = (df_is["preTaxProfit"] / df_is["revenue"]) * 100
    df_is["operating_profit_margin"] = df_is["operating_profit_margin"].replace([np.inf, -np.inf], 0).fillna(0)

    df_is["net_profit_margin"] = ((df_is["postTaxProfit"] / df_is["revenue"]) * 100)
    df_is["net_profit_margin"] = df_is["net_profit_margin"].replace([np.inf, -np.inf], 0).fillna(0)

    """
    Revenue Trend
    """
    # Plot asset history for all companies and overall yearly average
    fig_revenue = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_is['ticker'].unique():
        df_ticker = df_is[df_is['ticker'] == ticker].sort_values('year')
        fig_revenue.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['revenue'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_revenue_year_avg = df_is.groupby(['year'], as_index=False)['revenue'].mean()
    fig_revenue.add_trace(go.Scatter(
        x=df_revenue_year_avg['year'],
        y=df_revenue_year_avg['revenue'],
        mode='lines',
        name='Yearly Average Revenue (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_revenue.update_layout(
        title='Total Revenue Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Revenue',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_revenue_trend_html = fig_revenue.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Revenue Trend
    """

    """
    Gross Profit Trend
    """
    # Plot asset history for all companies and overall yearly average
    fig_gp = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_is['ticker'].unique():
        df_ticker = df_is[df_is['ticker'] == ticker].sort_values('year')
        fig_gp.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['preTaxProfit'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_gp_year_avg = df_is.groupby(['year'], as_index=False)['preTaxProfit'].mean()
    fig_gp.add_trace(go.Scatter(
        x=df_gp_year_avg['year'],
        y=df_gp_year_avg['preTaxProfit'],
        mode='lines',
        name='Yearly Average Gross Profit (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_gp.update_layout(
        title='Total Gross Profit Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Gross Profit',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_gp_trend_html = fig_gp.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Gross Profit Trend
    """

    """
    Net Profit Trend
    """
    fig_np = go.Figure()

    # Add traces for each company's asset trend
    for ticker in df_is['ticker'].unique():
        df_ticker = df_is[df_is['ticker'] == ticker].sort_values('year')
        fig_np.add_trace(go.Scatter(
            x=df_ticker['year'],
            y=df_ticker['postTaxProfit'],
            mode='lines+markers',
            name=f'{ticker}',
            line=dict(width=2),
            marker=dict(size=6),
            visible='legendonly',
        ))

    # Add overall yearly average
    df_np_year_avg = df_is.groupby(['year'], as_index=False)['postTaxProfit'].mean()
    fig_np.add_trace(go.Scatter(
        x=df_np_year_avg['year'],
        y=df_np_year_avg['postTaxProfit'],
        mode='lines',
        name='Yearly Average Net Profit (All Companies)',
        line=dict(width=3, dash='dash', color='black'),
        marker=dict(symbol='circle-open', size=8)
    ))

    fig_np.update_layout(
        title='Total Net Profit Trend - All Companies',
        xaxis_title='Year',
        yaxis_title='Net Profit',
        legend=dict(orientation='v'),
        width=1600,
        height=600,
        hovermode='x unified',
        showlegend=True  # Ensure the legend itself is visible
    )

    fig_np_trend_html = fig_np.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Net Profit Trend
    """

    """Gross Profit Margin"""
    # Calculate average currentRatio by symbol (across all years)
    avg_gpm_by_symbol = df_is.groupby('ticker')['gross_profit_margin'].mean().reset_index()
    avg_gpm_by_symbol.columns = ['ticker', 'avg_gross_profit_margin']
    avg_gpm_by_symbol = avg_gpm_by_symbol.sort_values('avg_gross_profit_margin', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_gpm = avg_gpm_by_symbol.nlargest(50, 'avg_gross_profit_margin')
    top_50_lowest_gpm = avg_gpm_by_symbol.nsmallest(50, 'avg_gross_profit_margin')

    top_50_highest_gpm_mean = top_50_highest_gpm['avg_gross_profit_margin'].mean()
    top_50_highest_gpm_median = top_50_highest_gpm['avg_gross_profit_margin'].median()
    top_50_highest_gpm_std = top_50_highest_gpm['avg_gross_profit_margin'].std()

    top_50_lowest_gpm_mean = top_50_lowest_gpm['avg_gross_profit_margin'].mean()
    top_50_lowest_gpm_median = top_50_lowest_gpm['avg_gross_profit_margin'].median()
    top_50_lowest_gpm_std = top_50_lowest_gpm['avg_gross_profit_margin'].std()

    # Create subplots: 1 row, 2 columns
    gpm_titles = [
        f'Top 50 Highest Average Gross Profit Margin - Mean: '
        f'{top_50_highest_gpm_mean:.3f}, Median: {top_50_highest_gpm_median:.3f}, Std: {top_50_highest_gpm_std:.3f}<b>',
        f'Top 50 Lowest Average Gross Profit Margin - Mean: '
        f'{top_50_lowest_gpm_mean:.3f}, Median: {top_50_lowest_gpm_median:.3f}, Std: {top_50_lowest_gpm_std:.3f}<b>',
    ]
    fig_gpm = make_subplots(
        rows=1, cols=2,
        subplot_titles=gpm_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig_gpm.add_trace(
        go.Bar(
            x=top_50_highest_gpm['avg_gross_profit_margin'],
            y=top_50_highest_gpm['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Gross Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig_gpm.add_trace(
        go.Bar(
            x=top_50_lowest_gpm['avg_gross_profit_margin'],
            y=top_50_lowest_gpm['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Gross Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig_gpm.update_layout(
        title_text="Distribution of Average Gross Profit Margin by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig_gpm.update_xaxes(title_text="Average Gross Profit Margin", row=1, col=1)
    fig_gpm.update_xaxes(title_text="Average Gross Profit Margin", row=1, col=2)
    fig_gpm.update_yaxes(title_text="Symbol", row=1, col=1)
    fig_gpm.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_gpm_html = fig_gpm.to_html(full_html=False, include_plotlyjs='cdn')
    """End Gross Profit Margin"""

    """Net Profit Margin"""
    # Calculate average currentRatio by symbol (across all years)
    avg_npm_by_symbol = df_is.groupby('ticker')['net_profit_margin'].mean().reset_index()
    avg_npm_by_symbol.columns = ['ticker', 'avg_net_profit_margin']
    avg_npm_by_symbol = avg_npm_by_symbol.sort_values('avg_net_profit_margin', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_npm = avg_npm_by_symbol.nlargest(50, 'avg_net_profit_margin')
    top_50_lowest_npm = avg_npm_by_symbol.nsmallest(50, 'avg_net_profit_margin')

    top_50_highest_npm_mean = top_50_highest_npm['avg_net_profit_margin'].mean()
    top_50_highest_npm_median = top_50_highest_npm['avg_net_profit_margin'].median()
    top_50_highest_npm_std = top_50_highest_npm['avg_net_profit_margin'].std()

    top_50_lowest_npm_mean = top_50_lowest_npm['avg_net_profit_margin'].mean()
    top_50_lowest_npm_median = top_50_lowest_npm['avg_net_profit_margin'].median()
    top_50_lowest_npm_std = top_50_lowest_npm['avg_net_profit_margin'].std()

    # Create subplots: 1 row, 2 columns
    npm_titles = [
        f'Top 50 Highest Average Net Profit Margin - Mean: '
        f'{top_50_highest_npm_mean:.3f}, Median: {top_50_highest_npm_median:.3f}, Std: {top_50_highest_npm_std:.3f}<b>',
        f'Top 50 Lowest Average Net Profit Margin - Mean: '
        f'{top_50_lowest_npm_mean:.3f}, Median: {top_50_lowest_npm_median:.3f}, Std: {top_50_lowest_npm_std:.3f}<b>',
    ]
    fig_npm = make_subplots(
        rows=1, cols=2,
        subplot_titles=npm_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig_npm.add_trace(
        go.Bar(
            x=top_50_highest_npm['avg_net_profit_margin'],
            y=top_50_highest_npm['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Net Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig_npm.add_trace(
        go.Bar(
            x=top_50_lowest_npm['avg_net_profit_margin'],
            y=top_50_lowest_npm['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Net Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig_npm.update_layout(
        title_text="Distribution of Average Net Profit Margin by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig_npm.update_xaxes(title_text="Average Net Profit Margin", row=1, col=1)
    fig_npm.update_xaxes(title_text="Average Net Profit Margin", row=1, col=2)
    fig_npm.update_yaxes(title_text="Symbol", row=1, col=1)
    fig_npm.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_npm_html = fig_npm.to_html(full_html=False, include_plotlyjs='cdn')
    """End Net Profit Margin"""

    """Operating Profit Margin"""
    # Calculate average currentRatio by symbol (across all years)
    avg_opm_by_symbol = df_is.groupby('ticker')['operating_profit_margin'].mean().reset_index()
    avg_opm_by_symbol.columns = ['ticker', 'avg_operating_profit_margin']
    avg_opm_by_symbol = avg_opm_by_symbol.sort_values('avg_operating_profit_margin', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_opm = avg_opm_by_symbol.nlargest(50, 'avg_operating_profit_margin')
    top_50_lowest_opm = avg_opm_by_symbol.nsmallest(50, 'avg_operating_profit_margin')

    top_50_highest_opm_mean = top_50_highest_opm['avg_operating_profit_margin'].mean()
    top_50_highest_opm_median = top_50_highest_opm['avg_operating_profit_margin'].median()
    top_50_highest_opm_std = top_50_highest_opm['avg_operating_profit_margin'].std()

    top_50_lowest_opm_mean = top_50_lowest_opm['avg_operating_profit_margin'].mean()
    top_50_lowest_opm_median = top_50_lowest_opm['avg_operating_profit_margin'].median()
    top_50_lowest_opm_std = top_50_lowest_opm['avg_operating_profit_margin'].std()

    # Create subplots: 1 row, 2 columns
    opm_titles = [
        f'Top 50 Highest Average Operating Profit Margin - Mean: '
        f'{top_50_highest_opm_mean:.3f}, Median: {top_50_highest_opm_median:.3f}, Std: {top_50_highest_opm_std:.3f}<b>',
        f'Top 50 Lowest Average Operating Profit Margin - Mean: '
        f'{top_50_lowest_opm_mean:.3f}, Median: {top_50_lowest_opm_median:.3f}, Std: {top_50_lowest_opm_std:.3f}<b>',
    ]
    fig_opm = make_subplots(
        rows=1, cols=2,
        subplot_titles=opm_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig_opm.add_trace(
        go.Bar(
            x=top_50_highest_opm['avg_operating_profit_margin'],
            y=top_50_highest_opm['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Operating Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig_opm.add_trace(
        go.Bar(
            x=top_50_lowest_opm['avg_operating_profit_margin'],
            y=top_50_lowest_opm['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Operating Profit Margin: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig_opm.update_layout(
        title_text="Distribution of Average Operating Profit Margin by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig_opm.update_xaxes(title_text="Average Operating Profit Margin", row=1, col=1)
    fig_opm.update_xaxes(title_text="Average Operating Profit Margin", row=1, col=2)
    fig_opm.update_yaxes(title_text="Symbol", row=1, col=1)
    fig_opm.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_opm_html = fig_opm.to_html(full_html=False, include_plotlyjs='cdn')
    """End Operating Profit Margin"""

    """
    Income Statement Indicator Correlation Metrix
    """
    gross_profit_margin_features = ["grossProfit", "revenue", "gross_profit_margin"]
    operating_profit_margin_features = ["preTaxProfit", "revenue", "operating_profit_margin"]
    net_profit_margin_cols = ["postTaxProfit", "revenue", "net_profit_margin"]

    # GPM
    # Filter data to include only rows with complete feature data
    df_corr_gpm = df_is[gross_profit_margin_features].copy()

    # Calculate correlation matrix
    correlation_matrix_gpm = df_corr_gpm.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_gpm = []
    for i in range(len(correlation_matrix_gpm)):
        for j in range(i + 1, len(correlation_matrix_gpm)):
            corr_value_cr = correlation_matrix_gpm.iloc[i, j]
            if not pd.isna(corr_value_cr):
                corr_values_gpm.append(corr_value_cr)

    mean_corr_gpm = np.mean(corr_values_gpm) if corr_values_gpm else 0
    median_corr_gpm = np.median(corr_values_gpm) if corr_values_gpm else 0
    std_corr_gpm = np.std(corr_values_gpm) if corr_values_gpm else 0

    # Create interactive heatmap using Plotly
    corr_fig_gpm = go.Figure()
    corr_fig_gpm = corr_fig_gpm.add_trace(go.Heatmap(
        z=correlation_matrix_gpm.values,
        x=correlation_matrix_gpm.columns,
        y=correlation_matrix_gpm.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_gpm.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_gpm.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Gross Profit Margin Indicator Features<br><sub>Mean: '
                    f'{mean_corr_gpm:.3f}, Median: {median_corr_gpm:.3f}, Std: {std_corr_gpm:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_gpm_metrix_html = corr_fig_gpm.to_html(full_html=False, include_plotlyjs='cdn')

    # Net Profit Margin
    df_corr_npm = df_is[net_profit_margin_cols].copy()

    # Calculate correlation matrix
    correlation_matrix_npm = df_corr_npm.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_npm = []
    for i in range(len(correlation_matrix_npm)):
        for j in range(i + 1, len(correlation_matrix_npm)):
            corr_value_npm = correlation_matrix_npm.iloc[i, j]
            if not pd.isna(corr_value_npm):
                corr_values_npm.append(corr_value_npm)

    mean_corr_npm = np.mean(corr_values_npm) if corr_values_npm else 0
    median_corr_npm = np.median(corr_values_npm) if corr_values_npm else 0
    std_corr_npm = np.std(corr_values_npm) if corr_values_npm else 0

    # Create interactive heatmap using Plotly
    corr_fig_npm = go.Figure()
    corr_fig_npm = corr_fig_npm.add_trace(go.Heatmap(
        z=correlation_matrix_npm.values,
        x=correlation_matrix_npm.columns,
        y=correlation_matrix_npm.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_npm.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_npm.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Net Profit Margin Indicator Features<br><sub>Mean: '
                    f'{mean_corr_npm:.3f}, Median: {median_corr_npm:.3f}, Std: {std_corr_npm:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_npm_metrix_html = corr_fig_npm.to_html(full_html=False, include_plotlyjs='cdn')

    # Net Profit Margin
    df_corr_opm = df_is[operating_profit_margin_features].copy()

    # Calculate correlation matrix
    correlation_matrix_opm = df_corr_opm.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_opm = []
    for i in range(len(correlation_matrix_opm)):
        for j in range(i + 1, len(correlation_matrix_opm)):
            corr_value_opm = correlation_matrix_opm.iloc[i, j]
            if not pd.isna(corr_value_opm):
                corr_values_opm.append(corr_value_opm)

    mean_corr_opm = np.mean(corr_values_opm) if corr_values_opm else 0
    median_corr_opm = np.median(corr_values_opm) if corr_values_opm else 0
    std_corr_opm = np.std(corr_values_opm) if corr_values_opm else 0

    # Create interactive heatmap using Plotly
    corr_fig_opm = go.Figure()
    corr_fig_opm.add_trace(go.Heatmap(
        z=correlation_matrix_opm.values,
        x=correlation_matrix_opm.columns,
        y=correlation_matrix_opm.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_opm.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_opm.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Operating Profit Margin Indicator Features<br><sub>Mean: '
                    f'{mean_corr_opm:.3f}, Median: {median_corr_opm:.3f}, Std: {std_corr_opm:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_opm_metrix_html = corr_fig_opm.to_html(full_html=False, include_plotlyjs='cdn')
    """End Income Statement Indicator"""

    context = {
        "request": request,
        "fig_corr_metrix_html": fig_corr_metrix_html,
        "fig_revenue_trend_html": fig_revenue_trend_html,
        "fig_gp_trend_html": fig_gp_trend_html,
        "fig_np_trend_html": fig_np_trend_html,
        "corr_fig_gpm_metrix_html": corr_fig_gpm_metrix_html,
        "corr_fig_npm_metrix_html": corr_fig_npm_metrix_html,
        "corr_fig_opm_metrix_html": corr_fig_opm_metrix_html,
        "fig_gpm_html": fig_gpm_html,
        "fig_npm_html": fig_npm_html,
        "fig_opm_html": fig_opm_html,
    }
    return templates.TemplateResponse("eda/income_statement.html", context=context)


@router.get("/cashflow", response_class=HTMLResponse)
async def eda_cashflow(request: Request, session: AsyncSession = Depends(session_manager.session)):
    df_cf = await get_cash_flow(session, yearly=True, year=2025, symbol="FPT")
    df_bs = await get_balance_sheet(session, yearly=True, year=2025, symbol="FPT")
    df_cf.fillna(value=0)
    df_bs.fillna(value=0)
    numeric_cols = df_cf.select_dtypes(include=[np.number]).columns.tolist()

    """
    Correlation Metrix
    """
    # Filter data to include only rows with complete feature data
    df_corr = df_cf[numeric_cols].copy()

    # Calculate correlation matrix
    correlation_matrix = df_corr.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values = []
    for i in range(len(correlation_matrix)):
        for j in range(i + 1, len(correlation_matrix)):
            corr_value = correlation_matrix.iloc[i, j]
            if not pd.isna(corr_value):
                corr_values.append(corr_value)

    mean_corr = np.mean(corr_values) if corr_values else 0
    median_corr = np.median(corr_values) if corr_values else 0
    std_corr = np.std(corr_values) if corr_values else 0

    # Create interactive heatmap using Plotly
    corr_fig = go.Figure()
    fig = corr_fig.add_trace(go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Cash Flow Features<br><sub>Mean: '
                    f'{mean_corr:.3f}, Median: {median_corr:.3f}, Std: {std_corr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    fig_corr_metrix_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Correlation Metrix
    """

    """
    Free Cash Flow
    """
    # Calculate average currentRatio by symbol (across all years)
    avg_fcf_by_symbol = df_cf.groupby('ticker')['freeCashFlow'].mean().reset_index()
    avg_fcf_by_symbol.columns = ['ticker', 'avg_freeCashFlow']
    avg_fcf_by_symbol = avg_fcf_by_symbol.sort_values('avg_freeCashFlow', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_fcf = avg_fcf_by_symbol.nlargest(50, 'avg_freeCashFlow')
    top_50_lowest_fcf = avg_fcf_by_symbol.nsmallest(50, 'avg_freeCashFlow')

    top_50_highest_fcf_mean = top_50_highest_fcf['avg_freeCashFlow'].mean()
    top_50_highest_fcf_median = top_50_highest_fcf['avg_freeCashFlow'].median()
    top_50_highest_fcf_std = top_50_highest_fcf['avg_freeCashFlow'].std()

    top_50_lowest_fcf_mean = top_50_lowest_fcf['avg_freeCashFlow'].mean()
    top_50_lowest_fcf_median = top_50_lowest_fcf['avg_freeCashFlow'].median()
    top_50_lowest_fcf_std = top_50_lowest_fcf['avg_freeCashFlow'].std()

    fcf_titles = [
        f'Top 50 Highest Average Free Cash Flow - Mean: '
        f'{top_50_highest_fcf_mean:.3f}, Median: {top_50_highest_fcf_median:.3f}, Std: {top_50_highest_fcf_std:.3f}<b>',
        f'Top 50 Lowest Average Free Cash Flow - Mean: '
        f'{top_50_lowest_fcf_mean:.3f}, Median: {top_50_lowest_fcf_median:.3f}, Std: {top_50_lowest_fcf_std:.3f}<b>',
    ]
    # Create subplots: 1 row, 2 columns
    fig_fcf = make_subplots(
        rows=1, cols=2,
        subplot_titles=fcf_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig_fcf.add_trace(
        go.Bar(
            x=top_50_highest_fcf['avg_freeCashFlow'],
            y=top_50_highest_fcf['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Free Cash Flow: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig_fcf.add_trace(
        go.Bar(
            x=top_50_lowest_fcf['avg_freeCashFlow'],
            y=top_50_lowest_fcf['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Free Cash Flow: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig_fcf.update_layout(
        title_text="Distribution of Average Free Cash Flow by Symbol (2000-2024)",
        height=800,
        width=1600,
        showlegend=False,
        hovermode='closest'
    )

    fig_fcf.update_xaxes(title_text="Average Free Cash Flow", row=1, col=1)
    fig_fcf.update_xaxes(title_text="Average Free Cash Flow", row=1, col=2)
    fig_fcf.update_yaxes(title_text="Symbol", row=1, col=1)
    fig_fcf.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_fcf_html = fig_fcf.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Free Cash Flow
    """

    """
    Free Cash Flow To Equity
    """
    df_cf["equity"] = df_bs["equity"]
    df_cf["freeCashFlowToEquity"] = df_cf["fromSale"] - df_cf["investCost"] + df_cf["equity"]

    # Calculate average currentRatio by symbol (across all years)
    avg_fcfe_by_symbol = df_cf.groupby('ticker')['freeCashFlowToEquity'].mean().reset_index()
    avg_fcfe_by_symbol.columns = ['ticker', 'avg_freeCashFlowToEquity']
    avg_fcfe_by_symbol = avg_fcfe_by_symbol.sort_values('avg_freeCashFlowToEquity', ascending=False)

    # Get top 50 highest and lowest
    top_50_highest_fcfe = avg_fcfe_by_symbol.nlargest(50, 'avg_freeCashFlowToEquity')
    top_50_lowest_fcfe = avg_fcfe_by_symbol.nsmallest(50, 'avg_freeCashFlowToEquity')

    top_50_highest_fcfe_mean = top_50_highest_fcfe['avg_freeCashFlowToEquity'].mean()
    top_50_highest_fcfe_median = top_50_highest_fcfe['avg_freeCashFlowToEquity'].median()
    top_50_highest_fcfe_std = top_50_highest_fcfe['avg_freeCashFlowToEquity'].std()

    top_50_lowest_fcfe_mean = top_50_lowest_fcfe['avg_freeCashFlowToEquity'].mean()
    top_50_lowest_fcfe_median = top_50_lowest_fcfe['avg_freeCashFlowToEquity'].median()
    top_50_lowest_fcfe_std = top_50_lowest_fcfe['avg_freeCashFlowToEquity'].std()

    fcfe_titles = [
        f'Top 50 Highest Average Free Cash Flow To Equity <br>Mean: '
        f'{top_50_highest_fcfe_mean:.3f}, Median: {top_50_highest_fcfe_median:.3f}, Std: {top_50_highest_fcfe_std:.3f}<b></br>',
        f'Top 50 Lowest Average Free Cash Flow To Equity <br>Mean: '
        f'{top_50_lowest_fcfe_mean:.3f}, Median: {top_50_lowest_fcfe_median:.3f}, Std: {top_50_lowest_fcfe_std:.3f}<b></br>',
    ]

    # Create subplots: 1 row, 2 columns
    fig_fcfe = make_subplots(
        rows=1, cols=2,
        subplot_titles=fcfe_titles,
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # Add top 50 highest chart
    fig_fcfe.add_trace(
        go.Bar(
            x=top_50_highest_fcfe['avg_freeCashFlowToEquity'],
            y=top_50_highest_fcfe['ticker'],
            orientation='h',
            marker=dict(color='green', opacity=0.7),
            name='Highest',
            hovertemplate='<b>%{y}</b><br>Avg Free Cash Flow To Equity: %{x:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add top 50 lowest chart
    fig_fcfe.add_trace(
        go.Bar(
            x=top_50_lowest_fcfe['avg_freeCashFlowToEquity'],
            y=top_50_lowest_fcfe['ticker'],
            orientation='h',
            marker=dict(color='red', opacity=0.7),
            name='Lowest',
            hovertemplate='<b>%{y}</b><br>Avg Free Cash Flow To Equity: %{x:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

    # Update layout
    fig_fcfe.update_layout(
        title_text="Distribution of Average Free Cash Flow To Equity by Symbol (2000-2024)",
        height=800,
        width=1400,
        showlegend=False,
        hovermode='closest'
    )

    fig_fcfe.update_xaxes(title_text="Average Free Cash Flow To Equity", row=1, col=1)
    fig_fcfe.update_xaxes(title_text="Average Free Cash Flow To Equity", row=1, col=2)
    fig_fcfe.update_yaxes(title_text="Symbol", row=1, col=1)
    fig_fcfe.update_yaxes(title_text="Symbol", row=1, col=2)

    fig_fcfe_html = fig_fcfe.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Free Cash Flow To Equity
    """

    """Free Cash Flow To Equity Correlation"""
    # Free Cash Flow To Equity
    fcfe_features = ["fromSale", "investCost", "equity", "freeCashFlowToEquity"]
    df_corr_fcfe = df_cf[fcfe_features].copy()

    # Calculate correlation matrix
    correlation_matrix_fcfe = df_corr_fcfe.corr()

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values_fcfe = []
    for i in range(len(correlation_matrix_fcfe)):
        for j in range(i + 1, len(correlation_matrix_fcfe)):
            corr_value_fcfe = correlation_matrix_fcfe.iloc[i, j]
            if not pd.isna(corr_value_fcfe):
                corr_values_fcfe.append(corr_value_fcfe)

    mean_corr_fcfe= np.mean(corr_values_fcfe) if corr_values_fcfe else 0
    median_corr_fcfe = np.median(corr_values_fcfe) if corr_values_fcfe else 0
    std_corr_fcfe = np.std(corr_values_fcfe) if corr_values_fcfe else 0

    # Create interactive heatmap using Plotly
    corr_fig_fcfe = go.Figure()
    corr_fig_fcfe.add_trace(go.Heatmap(
        z=correlation_matrix_fcfe.values,
        x=correlation_matrix_fcfe.columns,
        y=correlation_matrix_fcfe.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix_fcfe.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
    ))

    corr_fig_fcfe.update_layout(
        title={
            'text': f'<b>Correlation Matrix - Free Cash Flow To Equity Indicator Features<br><sub>Mean: '
                    f'{mean_corr_fcfe:.3f}, Median: {median_corr_fcfe:.3f}, Std: {std_corr_fcfe:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
        xaxis={'side': 'bottom'},
    )

    corr_fig_fcfe_metrix_html = corr_fig_fcfe.to_html(full_html=False, include_plotlyjs='cdn')
    """End Free Cash Flow To Equity Correlation"""

    context = {
        "request": request,
        "fig_corr_metrix_html": fig_corr_metrix_html,
        "corr_fig_fcfe_metrix_html": corr_fig_fcfe_metrix_html,
        "fig_fcf_html": fig_fcf_html,
        "fig_fcfe_html": fig_fcfe_html,
    }
    return templates.TemplateResponse("eda/cashflow.html", context=context)