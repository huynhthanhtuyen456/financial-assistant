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


@router.get("/balance-sheet/", response_class=HTMLResponse)
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


@router.get("/distribution/", response_class=HTMLResponse)
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