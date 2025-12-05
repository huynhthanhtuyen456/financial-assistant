import base64
import io

import matplotlib
import plotly.graph_objects as go
import requests
from tensorflow.python.keras.losses import mean_squared_error

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.financial_ratio import FinancialRatio
from fastapi import APIRouter, Query, Depends, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.balancesheet import BalanceSheet
from db import session_manager
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
import os
from core.helpers import get_balance_sheet


router = APIRouter(
    prefix="/eda",
    tags=["eda-dashboard"]
)
templates = Jinja2Templates(directory="templates")


@router.get("/balance-sheet/", response_class=HTMLResponse)
async def index(request: Request, session: AsyncSession = Depends(session_manager.session),):
    df_bs = await get_balance_sheet(session, yearly=True)
    numeric_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
    df_bs.sort_values(by=['ticker', 'year'], inplace=True)
    df_bs.reset_index(drop=True, inplace=True)
    # Melt the dataframe to long format for easier plotting
    melted = df_bs[numeric_cols].melt(var_name='variable', value_name='value')

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
                           height=600, width=1200,
                           points='outliers')

    fig_violin.update_layout(xaxis_tickangle=-45)
    fig_violin.show()

    # Time series trend: Average value per year for each feature
    trend_by_year = df_bs[numeric_cols].groupby(df_bs['year']).mean()
    trend_melted = trend_by_year.melt(id_vars='year', var_name='feature', value_name='avg_value')

    fig_trend = px.line(trend_melted, x='year', y='avg_value', color='feature',
                        title='Trend of Features Over Time (Average per Year)',
                        labels={'year': 'Year', 'avg_value': 'Average Value', 'feature': 'Feature'},
                        height=600, width=1200)

    fig_trend_html = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')
    """Create distribution plots for each numeric feature"""

    # context var
    context = {
        "fig_trend_html": fig_trend_html
    }

    return templates.TemplateResponse("eda/balance_sheet.html", {"request": request, "context": context})