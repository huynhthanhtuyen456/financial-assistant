"""
Core Helper Functions for Financial Machine Learning.

This module contains utility functions for:
1.  **Data Preprocessing:** Transforming time-series data into supervised learning sequences.
2.  **Model Management:** Building, training, saving, and loading Keras (LSTM/RNN) and
    Scikit-Learn (Random Forest/Linear Regression) models.
3.  **Data Retrieval:** Asynchronous database fetchers for Balance Sheets, Income
    Statements, and Cash Flows using SQLAlchemy.
4.  **Visualization:** Generating Plotly HTML snippets for financial trend analysis.
"""

import os

import joblib
import math
import numpy as np
import pandas as pd
from fastapi import HTTPException
from plotly import graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, LeakyReLU, BatchNormalization, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

from models.balancesheet import BalanceSheet
from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.stock import Stock


def create_sequences(dataset: np.ndarray, look_back_years: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a time-series dataset into a supervised learning format using a sliding window.

    This function creates 'X' (features) and 'y' (target) sets. 'X' consists of
    sequences of length `look_back_years`, and 'y' is the value immediately
    following that sequence.



    Args:
        dataset (np.ndarray): The scaled time-series data.
        look_back_years (int): The number of past time steps to use for prediction.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the input sequences (X)
        and the target values (y).
    """
    x, y = [], []
    for i in range(len(dataset) - look_back_years):
        x.append(dataset[i:(i + look_back_years)])
        y.append(dataset[i + look_back_years])
    return np.array(x), np.array(y)


def get_lstm_model(
        model_name: str,
        n_inputs: int,
        n_features: list,
        x_train: np.ndarray,
        y_train: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray
) -> Sequential:
    """
    Constructs, trains, or loads a Hybrid LSTM-GRU Neural Network.

    If a model file exists at `model_name`, it is loaded. Otherwise, a new model
    is built using a specific architecture optimized for financial time series:
    Input -> LSTM -> LeakyReLU -> GRU -> Dropout -> BatchNorm -> LSTM -> Output.



[Image of LSTM neural network architecture]


    Args:
        model_name (str): Path to save/load the .keras model file.
        n_inputs (int): Time steps (sequence length).
        n_features (list): List of feature names (used to determine input dimension).
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        val_x (np.ndarray): Validation features.
        val_y (np.ndarray): Validation targets.

    Returns:
        Sequential: The trained Keras model.
    """
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # Build Model Architecture
        model = Sequential([
            Input(shape=(n_inputs, len(n_features))),
            # First layer: LSTM to capture long-term dependencies
            LSTM(64, activation='relu', return_sequences=True),
            LeakyReLU(),
            # Second layer: GRU for efficient computation of temporal features
            GRU(50, activation='relu', return_sequences=True),
            Dropout(0.5),  # Regularization to prevent overfitting
            BatchNormalization(),  # Stabilize learning
            # Third layer: LSTM to condense features
            LSTM(32, activation='relu'),
            LeakyReLU(),
            Dropout(0.5),
            # Output layer
            Dense(len(n_features))
        ])

        model.compile(loss='mean_squared_error', optimizer='adam')

        # Callbacks for optimal training
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', save_best_only=True)
        reduce = ReduceLROnPlateau(patience=100, factor=0.5)

        model.fit(
            x_train, y_train,
            validation_data=(val_x, val_y),
            epochs=int(math.ceil(n_inputs / len(n_features))),
            batch_size=32,
            verbose=0,
            callbacks=[early_stopping, mc, reduce]
        )

        # Ensure directory exists and save
        os.makedirs("models", exist_ok=True)
        model.save(model_name)

    return model


def get_rnn_model(
        model_name: str,
        n_inputs: int,
        n_features: list,
        x_train: np.ndarray,
        y_train: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray
) -> Sequential:
    """
    Constructs, trains, or loads a Simple Recurrent Neural Network (RNN).

    This model uses `SimpleRNN` layers. While less powerful than LSTM for long
    sequences, it can be effective for shorter financial trends with less computational cost.



    Args:
        model_name (str): Path to save/load the .keras model file.
        n_inputs (int): Time steps (sequence length).
        n_features (list): List of feature names.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        val_x (np.ndarray): Validation features.
        val_y (np.ndarray): Validation targets.

    Returns:
        Sequential: The trained Keras RNN model.
    """
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # Build Model Architecture
        model = Sequential([
            Input(shape=(n_inputs, len(n_features))),
            SimpleRNN(120, activation='relu', return_sequences=True),
            SimpleRNN(120, return_sequences=False),
            Dense(len(n_features))
        ])

        model.compile(loss='mean_squared_error', optimizer='adam')

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, mode='min')
        mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

        model.fit(
            x_train, y_train,
            validation_data=(val_x, val_y),
            epochs=100,
            batch_size=16,
            verbose=0,
            callbacks=[early_stopping, mc]
        )

        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(model_name)

    return model


def get_random_forest_model(
        model_name: str,
        n_inputs: int,
        n_features: list,
        x_train: np.ndarray,
        y_train: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray
) -> RandomForestRegressor:
    """
    Constructs or loads a Random Forest Regressor.



[Image of Random Forest algorithm structure]


    Note: Random Forest requires 2D input (samples, features). If `x_train` is
    3D (from time-series sequencing), it is flattened before training.

    Args:
        model_name (str): Path to save/load the .joblib model file.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.

    Returns:
        RandomForestRegressor: The trained Scikit-Learn model.
    """
    if os.path.exists(model_name):
        model = joblib.load(model_name)
    else:
        # Reshape input data for Random Forest (Flatten time steps)
        x_train_reshaped = x_train.reshape(x_train.shape[0], -1)

        # Build and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train_reshaped, y_train)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_name)

    return model


def get_mlr_model(
        model_name: str,
        n_inputs: int,
        n_features: list,
        x_train: np.ndarray,
        y_train: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray
) -> LinearRegression:
    """
    Constructs or loads a Multiple Linear Regression model.



[Image of Linear Regression best fit line]


    Similar to Random Forest, this flattens 3D time-series inputs into 2D
    before fitting.

    Args:
        model_name (str): Path to save/load the .joblib model file.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.

    Returns:
        LinearRegression: The trained Scikit-Learn model.
    """
    if os.path.exists(model_name):
        model = joblib.load(model_name)
    else:
        # Reshape input data for Multiple Linear Regression
        x_train_reshaped = x_train.reshape(x_train.shape[0], -1)

        # Build and train model
        model = LinearRegression()
        model.fit(x_train_reshaped, y_train)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_name)

    return model


async def get_symbols(session: AsyncSession) -> list[dict]:
    """
    Fetches the list of all available stock symbols from the database.

    Args:
        session (AsyncSession): The database session.

    Returns:
        list[dict]: A list of dictionaries containing 'symbol' and 'eng_name'.

    Raises:
        HTTPException: 404 if no stocks are found.
    """
    stmt_st = select(Stock).order_by(Stock.symbol)
    queryset_stock = await session.execute(stmt_st)
    stock_data = queryset_stock.fetchall()

    if not stock_data:
        raise HTTPException(status_code=404, detail="No stocks found.")

    stocks = []
    for row in stock_data:
        item = row[0].__dict__
        stocks.append({"symbol": item["symbol"], "eng_name": item["eng_name"]})

    return stocks


async def get_balance_sheet(
        session: AsyncSession,
        symbol: str = "FPT",
        year: int = 2025,
        yearly: bool = True
) -> pd.DataFrame:
    """
    Fetches Balance Sheet data from the database and returns it as a DataFrame.

    Args:
        session (AsyncSession): The database session.
        symbol (str): Stock ticker (unused in query currently, filters applied later).
        year (int): Target year (unused in query currently).
        yearly (bool): Filter by yearly reports (True) or quarterly (False).

    Returns:
        pd.DataFrame: DataFrame containing flattened balance sheet JSON data.
    """
    stmt_bs = (select(BalanceSheet).where(BalanceSheet.yearly == yearly))
    queryset_bs = await session.execute(stmt_bs)
    balance_sheet_data = queryset_bs.fetchall()

    if not balance_sheet_data:
        raise HTTPException(status_code=404, detail="Balance Sheet data not found.")

    bs_data = []
    for row in balance_sheet_data:
        item = row[0].__dict__
        bs_data.extend(item["balance_sheet"])

    df_balance_sheet = pd.DataFrame(bs_data)
    return df_balance_sheet


async def get_income_statement(
        session: AsyncSession,
        symbol: str,
        year: int,
        yearly: bool = True
) -> pd.DataFrame:
    """
    Fetches Income Statement data from the database.

    Args:
        session (AsyncSession): The database session.
        yearly (bool): Filter by yearly reports (True) or quarterly (False).

    Returns:
        pd.DataFrame: DataFrame containing flattened income statement JSON data.
    """
    stmt_bs = (select(IncomeStatement).where(IncomeStatement.yearly == True))
    queryset_bs = await session.execute(stmt_bs)
    income_data = queryset_bs.fetchall()

    if not income_data:
        raise HTTPException(status_code=404, detail="Income Statement data not found.")

    is_data = []
    for row in income_data:
        item = row[0].__dict__
        is_data.extend(item["income_statement"])

    df_income = pd.DataFrame(is_data)
    return df_income


async def get_cash_flow(
        session: AsyncSession,
        symbol: str,
        year: int,
        yearly: bool = True
) -> pd.DataFrame:
    """
    Fetches Cash Flow data from the database.

    Args:
        session (AsyncSession): The database session.
        yearly (bool): Filter by yearly reports (True) or quarterly (False).

    Returns:
        pd.DataFrame: DataFrame containing flattened cash flow JSON data.
    """
    stmt_bs = (select(Cashflow).where(Cashflow.yearly == True))
    queryset_bs = await session.execute(stmt_bs)
    cashflow_data = queryset_bs.fetchall()

    if not cashflow_data:
        raise HTTPException(status_code=404, detail="Cash Flow data not found.")

    cf_data = []
    for row in cashflow_data:
        item = row[0].__dict__
        cf_data.extend(item["cashflow"])

    df_cashflow = pd.DataFrame(cf_data)
    return df_cashflow


def train_and_predict_ratio_random_forest(
        df_balance_sheet: pd.DataFrame,
        symbol: str,
        features: list,
        target_col: str = 'currentRatio',
        prediction_year: int = 2025,
) -> str:
    """
    Pipeline to train a Random Forest Regressor and visualize predictions.

    This function isolates a specific ticker, trains a Random Forest model on
    selected features to predict a target ratio, and generates a Plotly chart.

    Pipeline Steps:
    1.  Filter Data by Symbol.
    2.  Clean NaNs and Infinite values.
    3.  Encode categorical Symbol (if mixed) and prepare Feature/Target arrays.
    4.  Train Random Forest on historical data.
    5.  Forecast the target ratio for the `prediction_year`.
    6.  Generate Plotly HTML.

    Args:
        df_balance_sheet (pd.DataFrame): Source financial data.
        symbol (str): Ticker symbol to analyze.
        features (list): List of column names to use as input features.
        target_col (str): The column name to predict (e.g., 'currentRatio').
        prediction_year (int): The future year to predict.

    Returns:
        str: HTML string containing the Plotly interactive chart.
    """
    # 1. Prepare data
    df = df_balance_sheet[df_balance_sheet["ticker"] == symbol].copy()
    if df.empty:
        raise ValueError(f"Symbol '{symbol}' not found in the DataFrame.")

    numeric_cols = df_balance_sheet.select_dtypes(include=[np.number]).columns.tolist()

    # Clean data (In-place replacement)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)

    le = LabelEncoder()
    df['symbol_enc'] = le.fit_transform(df['ticker'].astype(str))

    # 2. Define features and target
    feature_cols = features + ['symbol_enc']
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    # 3. Train Random Forest (Full dataset for maximum context)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 4. Generate In-Sample Predictions (History)
    df_sorted = df.sort_values('year').reset_index(drop=True)
    X_sorted = df_sorted[feature_cols].astype(float)
    in_sample_predictions = rf.predict(X_sorted)

    # 5. Predict Next Year
    # Logic: Take the most recent year's data, increment year, use as input
    last_year_data = df_sorted[df_sorted["year"] == prediction_year - 1]

    # Fallback if specific prior year missing, take max year
    if last_year_data.empty:
        last_year_data = df_sorted.iloc[[-1]]

    next_row = last_year_data.copy()
    next_row['year'] = prediction_year  # Set the prediction year
    next_row['symbol_enc'] = 0  # Encoder fitted on single symbol = 0

    X_next = next_row[feature_cols].astype(float).values.reshape(1, -1)
    # pred_next = float(rf.predict(X_next)[0])  # (Prediction calculated but not plotted in final trace)

    # 6. Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted['year'], y=df_sorted[target_col],
        mode='lines+markers', name=f'{symbol} History',
        line=dict(width=2), marker=dict(size=6)
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted['year'], y=in_sample_predictions,
        mode='lines', name='RF In-Sample Fit',
        line=dict(width=2, dash='dot')
    ))

    fig.update_layout(
        title=f'Random Forest: {symbol} {target_col} (Historical vs Fit)',
        xaxis_title='Year',
        yaxis_title=target_col,
        width=900,
        height=500,
        legend_title_text='Legend'
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return fig_html


def train_and_predict_ratio_linear_regression(
        df: pd.DataFrame,
        symbol: str,
        target_col: str,
        prediction_year: int,
        title_suffix: str
) -> str:
    """
    Trains a simple Linear Regression model to forecast a trend line based solely on 'Year'.

    This provides a baseline trend (increasing/decreasing) comparison against
    more complex models.



[Image of Linear Regression best fit line]


    Args:
        df (pd.DataFrame): Source data.
        symbol (str): Ticker symbol.
        target_col (str): The column to trend.
        prediction_year (int): The year to forecast.
        title_suffix (str): Description for the chart title (e.g., "Current Ratio").

    Returns:
        str: HTML string containing the Plotly interactive chart.
    """
    # Filter and prepare data
    df_symbol = df[df['ticker'] == symbol].sort_values('year').copy()
    df_symbol = df_symbol.dropna(subset=[target_col]).reset_index(drop=True)

    if len(df_symbol) < 2:
        return "<div>Insufficient data for Linear Regression</div>"

    # Feature: Year, Target: Ratio
    X = df_symbol[['year']].values
    y = df_symbol[target_col].values

    # Train Model
    lr = LinearRegression()
    lr.fit(X, y)

    # Predict Trend
    y_trend = lr.predict(X)

    # Predict Future
    future_X = np.array([[prediction_year]])
    future_pred = lr.predict(future_X)[0]

    # Assess Trend Direction
    slope = lr.coef_[0]
    trend_desc = "Increasing" if slope > 0 else "Decreasing"

    # Visualization
    fig = go.Figure()

    # 1. Historical Actuals
    fig.add_trace(go.Scatter(
        x=df_symbol['year'], y=y,
        mode='markers', name='Actual History',
        marker=dict(color='blue', size=8)
    ))

    # 2. Linear Trend Line
    fig.add_trace(go.Scatter(
        x=df_symbol['year'], y=y_trend,
        mode='lines', name='Linear Trend',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # 3. Future Prediction
    fig.add_trace(go.Scatter(
        x=[prediction_year], y=[future_pred],
        mode='markers+text', name=f'LR Prediction {prediction_year}',
        marker=dict(color='red', size=12, symbol='star'),
        text=[f'{future_pred:.2f}'], textposition="top center"
    ))

    fig.update_layout(
        title=f"Linear Regression Trend - {title_suffix} ({trend_desc})",
        xaxis_title="Year",
        yaxis_title=target_col,
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')