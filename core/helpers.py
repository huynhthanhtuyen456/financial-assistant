import os

import math
import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, LeakyReLU, BatchNormalization, SimpleRNN
from tensorflow.keras.models import Sequential, load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from plotly import graph_objs as go

from models.balancesheet import BalanceSheet
from models.cashflow import Cashflow
from models.income_statement import IncomeStatement
from models.stock import Stock


def create_sequences(dataset, look_back_years):
    x, y = [], []
    for i in range(len(dataset) - look_back_years):
        x.append(dataset[i:(i + look_back_years)])
        y.append(dataset[i + look_back_years])
    return np.array(x), np.array(y)


def get_lstm_model(model_name, n_inputs, n_features, x_train, y_train, val_x, val_y):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # Build and train model
        model = Sequential([
            Input(shape=(n_inputs, len(n_features))),
            LSTM(64, activation='relu', return_sequences=True),
            LeakyReLU(),
            GRU(50, activation='relu', return_sequences=True),
            Dropout(0.5),
            BatchNormalization(),  # Batch Normalization layer
            LSTM(32, activation='relu'),
            LeakyReLU(),
            Dropout(0.5),
            Dense(len(n_features))
        ])

        model.compile(loss='mean_squared_error', optimizer='adam')

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

        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(model_name)

    return model


def get_rnn_model(model_name, n_inputs, n_features, x_train, y_train, val_x, val_y):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # Build and train model
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


def get_random_forest_model(model_name, n_inputs, n_features, x_train, y_train, val_x, val_y):
    if os.path.exists(model_name):
        model = joblib.load(model_name)
    else:
        # Reshape input data for Random Forest
        x_train_reshaped = x_train.reshape(x_train.shape[0], -1)

        # Build and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train_reshaped, y_train)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_name)

    return model


def get_mlr_model(model_name, n_inputs, n_features, x_train, y_train, val_x, val_y):
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


async def get_symbols(session: AsyncSession):
    # Get balance sheet data
    stmt_st = select(Stock).order_by(Stock.symbol)
    queryset_stock = await session.execute(stmt_st)
    stock_data = queryset_stock.fetchall()

    if not stock_data:
        raise HTTPException(status_code=404)

    # Extract data
    stocks = []
    for row in stock_data:
        item = row[0].__dict__
        stocks.append({"symbol": item["symbol"], "eng_name": item["eng_name"]})

    return stocks


async def get_balance_sheet(session: AsyncSession, symbol: str = "FPT", year: int = 2025, yearly: bool = True):
    stmt_bs = (select(BalanceSheet).where(BalanceSheet.yearly == yearly))
    queryset_bs = await session.execute(stmt_bs)
    balance_sheet_data = queryset_bs.fetchall()

    if not balance_sheet_data:
        raise HTTPException(status_code=404)

    # Extract data
    bs_data = []
    for row in balance_sheet_data:
        item = row[0].__dict__
        bs_data.extend(item["balance_sheet"])

    df_balance_sheet = pd.DataFrame(bs_data)
    return df_balance_sheet


async def get_income_statement(session: AsyncSession, symbol: str, year: int, yearly: bool = True):
    stmt_bs = (select(IncomeStatement)
               .where(IncomeStatement.yearly == True))
    queryset_bs = await session.execute(stmt_bs)
    income_data = queryset_bs.fetchall()

    if not income_data:
        raise HTTPException(status_code=404)

    # Extract data
    is_data = []
    for row in income_data:
        item = row[0].__dict__
        is_data.extend(item["income_statement"])

    df_income = pd.DataFrame(is_data)
    return df_income


async def get_cash_flow(session: AsyncSession, symbol: str, year: int, yearly: bool = True):
    stmt_bs = (select(Cashflow)
               .where(Cashflow.yearly == True))
    queryset_bs = await session.execute(stmt_bs)
    cashflow_data = queryset_bs.fetchall()

    if not cashflow_data:
        raise HTTPException(status_code=404)

    # Extract data
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
):
    """
    Trains a Random Forest model to predict a financial ratio for a given stock symbol.

    This function performs the following steps:
    1. Filters the data for the specified symbol.
    2. Prepares features (numeric columns + encoded symbol) and the target variable.
    3. Splits the data and trains a RandomForestRegressor.
    4. Calculates and ranks feature importances.
    5. Predicts on historical data and for the next year.
    6. Generates an interactive Plotly chart visualizing the results.

    Args:
        df_balance_sheet (pd.DataFrame): The main DataFrame containing balance sheet data for multiple tickers.
                                         Must include 'ticker', 'year', and other numeric financial columns.
        symbol (str): The stock ticker symbol to model (e.g., 'AAPL').
        target_col (str): The name of the column to be predicted. Defaults to 'currentRatio'.

    Returns:
        dict: A dictionary containing:
              - 'plot_html': A string of HTML for the Plotly visualization.
              - 'feature_importances': A list of tuples (feature, importance score).
              - 'next_year_prediction': The predicted value for the next year.
    """
    # 1. Prepare data: use numeric columns + encoded symbol for the given ticker
    df = df_balance_sheet[df_balance_sheet["ticker"] == symbol].copy()
    if df.empty:
        raise ValueError(f"Symbol '{symbol}' not found in the DataFrame.")

    numeric_cols = df_balance_sheet.select_dtypes(include=[np.number]).columns.tolist()

    # --- Code Improvement: The original replace was not in-place ---
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
    df.fillna(0, inplace=True)  # General fillna for simplicity

    le = LabelEncoder()
    df['symbol_enc'] = le.fit_transform(df['ticker'].astype(str))

    # 2. Define features and target
    # feature_cols = [c for c in numeric_cols if c != target_col] + ['symbol_enc']
    feature_cols = features + ['symbol_enc']
    X = df[feature_cols].astype(float)
    y = df[target_col].astype(float)

    # 3. Train / test split for model evaluation (optional, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 4. Train Random Forest on the full dataset for the best prediction
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)  # Train on all data for the symbol for final prediction

    # 5. Get feature importances
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    # 6. Generate predictions
    df_sorted = df.sort_values('year').reset_index(drop=True)
    X_sorted = df_sorted[feature_cols].astype(float)
    in_sample_predictions = rf.predict(X_sorted)

    # 7. Predict next year
    # --- Code Improvement: Avoid hardcoding the year '2023' ---
    last_year_data = df_sorted[df_sorted["year"] == prediction_year - 1]
    next_year = int(last_year_data['year']) + 1

    next_row = last_year_data.copy()
    next_row['year'] = next_year
    # The label encoder was fit on a single symbol, so its transform will be [0]
    next_row['symbol_enc'] = 0

    X_next = next_row[feature_cols].astype(float).values.reshape(1, -1)
    pred_next = float(rf.predict(X_next)[0])

    # 8. Plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['year'], y=df_sorted[target_col],
                             mode='lines+markers', name=f'{symbol} History',
                             line=dict(width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=df_sorted['year'], y=in_sample_predictions,
                             mode='lines', name='In-Sample Prediction',
                             line=dict(width=2, dash='dot')))
    # fig.add_trace(go.Scatter(x=[next_year], y=[pred_next],
    #                          mode='markers', name=f'Prediction for {next_year}',
    #                          marker=dict(symbol='star', size=15, color='red')))

    fig.update_layout(
        title=f'Random Forest: {symbol} {target_col} (Historical & Predicted)',
        xaxis_title='Year',
        yaxis_title=target_col,
        width=900,
        height=500,
        legend_title_text='Legend'
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return fig_html


def train_and_predict_ratio_linear_regression(df, symbol, target_col, prediction_year, title_suffix):
    """
    Trains a Linear Regression model using 'Year' as the feature to project a trend line.
    Returns the Plotly HTML.
    """
    # Filter and prepare data
    df_symbol = df[df['ticker'] == symbol].sort_values('year').copy()
    df_symbol = df_symbol.dropna(subset=[target_col]).reset_index(drop=True)

    if len(df_symbol) < 2:
        return "<div>Insufficient data for Linear Regression</div>"

    # We use YEAR as the feature for forecasting the trend
    X = df_symbol[['year']].values
    y = df_symbol[target_col].values

    # Train Model
    lr = LinearRegression()
    lr.fit(X, y)

    # Predict History (for the trend line)
    y_trend = lr.predict(X)

    # Predict Future
    future_X = np.array([[prediction_year]])
    future_pred = lr.predict(future_X)[0]

    # Calculate Assessment (generic logic based on ratio direction)
    slope = lr.coef_[0]
    trend_desc = "Increasing" if slope > 0 else "Decreasing"

    # Create Visualization
    fig = go.Figure()

    # 1. Historical Actuals
    fig.add_trace(go.Scatter(
        x=df_symbol['year'],
        y=y,
        mode='markers',
        name='Actual History',
        marker=dict(color='blue', size=8)
    ))

    # 2. Linear Trend Line
    fig.add_trace(go.Scatter(
        x=df_symbol['year'],
        y=y_trend,
        mode='lines',
        name='Linear Trend',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # 3. Future Prediction
    fig.add_trace(go.Scatter(
        x=[prediction_year],
        y=[future_pred],
        mode='markers+text',
        name=f'LR Prediction {prediction_year}',
        marker=dict(color='red', size=12, symbol='star'),
        text=[f'{future_pred:.2f}'],
        textposition="top center"
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
