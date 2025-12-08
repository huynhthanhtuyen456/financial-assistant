import os

import math
import numpy as np
from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# ML/DL Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.ext.asyncio import AsyncSession
from tensorflow.keras.models import load_model

# Project Modules
from core.helpers import get_balance_sheet, create_sequences, get_symbols
from db import session_manager

router = APIRouter(
    prefix="/models",
    tags=["models"]
)

templates = Jinja2Templates(directory="templates")


# ==========================================
# 1. HELPER FUNCTIONS & METRICS
# ==========================================

def calculate_metrics(y_true, y_pred, model_name):
    """Helper to calculate regression metrics."""
    # Ensure inputs are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Handle edge case where predictions are empty or NaN
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "name": model_name,
            "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0
        }

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "name": model_name,
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }


# ==========================================
# 2. MODEL SPECIFIC LOGIC
# ==========================================

def run_dl_model(df, symbol, model_type="LSTM"):
    """
    Loads saved LSTM/RNN models (which predict Balance Sheet components),
    derives the Current Ratio from predictions, and compares to actuals.
    """
    # Features used in balancesheet.py for Current Ratio prediction
    features = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
                'otherDebt', 'payable', 'shortDebt', 'longDebt']

    # Ensure data is sorted
    df = df.sort_values('year')
    df = df.dropna(subset=features).reset_index(drop=True)

    if len(df) < 5:
        return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    # Data Prep
    dataset = df[features].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on available data to match input distribution expected by model
    scaler.fit(dataset)
    scaled_data = scaler.transform(dataset)

    # Sequence Creation
    look_back = 3
    X, y = create_sequences(scaled_data, look_back)

    if len(X) == 0:
        return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    # Split into Train/Test (Last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Load Model
    if model_type == "LSTM":
        model_path = f"trained_models/current_ratio_{symbol}.keras"
    else:
        # Assuming RNN model naming convention follows
        model_path = f"trained_models/rnn_current_ratio_{symbol}.keras"

    if not os.path.exists(model_path):
        return {"name": f"{model_type} (Not Found)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    try:
        model = load_model(model_path)
    except Exception as e:
        return {"name": f"{model_type} (Error)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    # Predict
    # X_test shape: (samples, look_back, features)
    test_predict_scaled = model.predict(X_test, verbose=0)

    # Inverse transform to get component values
    # y_test and predictions are both (samples, 8)
    pred_inv = scaler.inverse_transform(test_predict_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate Current Ratio from components
    # Indices based on features list:
    # Assets: cash(0), shortReceivable(1), inventory(2), shortAsset(3)
    # Liabs: otherDebt(4), payable(5), shortDebt(6), longDebt(7)

    # Predicted Ratio
    pred_current_assets = np.sum(pred_inv[:, 0:4], axis=1)
    pred_current_liabs = np.sum(pred_inv[:, 4:8], axis=1)  # Sum of otherDebt, payable, shortDebt, longDebt

    # Avoid division by zero
    pred_ratio = np.divide(pred_current_assets, pred_current_liabs,
                           out=np.zeros_like(pred_current_assets),
                           where=pred_current_liabs != 0)

    # Actual Ratio (Derived from inverse transformed actuals to maintain consistency)
    true_current_assets = np.sum(y_test_inv[:, 0:4], axis=1)
    true_current_liabs = np.sum(y_test_inv[:, 4:8], axis=1)

    true_ratio = np.divide(true_current_assets, true_current_liabs,
                           out=np.zeros_like(true_current_assets),
                           where=true_current_liabs != 0)

    return calculate_metrics(true_ratio, pred_ratio, f"{model_type} (Current Ratio)")


def run_random_forest(df, target_col="currentRatio"):
    # Features used for regression
    # We use lags of the components + previous ratio
    component_cols = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
                      'otherDebt', 'payable', 'shortDebt', 'longDebt']

    df_rf = df.copy()

    # Calculate target if not present
    if target_col not in df_rf.columns:
        current_assets = df_rf['cash'] + df_rf['shortReceivable'] + df_rf['inventory'] + df_rf['shortAsset']
        current_liabs = df_rf['payable'] + df_rf['shortDebt'] + df_rf['otherDebt'] + df_rf['longDebt']
        df_rf[target_col] = current_assets / current_liabs
        df_rf[target_col] = df_rf[target_col].replace([np.inf, -np.inf], 0).fillna(0)

    # Create lag features
    feature_cols = []
    for col in component_cols + [target_col]:
        df_rf[f'{col}_prev'] = df_rf[col].shift(1)
        feature_cols.append(f'{col}_prev')

    df_rf = df_rf.dropna()

    X = df_rf[feature_cols].values
    y = df_rf[target_col].values

    if len(X) < 5:
        return {"name": "Random Forest", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    # Time series split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return calculate_metrics(y_test, y_pred, "Random Forest")


def run_linear_regression(df, target_col="currentRatio"):
    # Same feature prep as RF
    component_cols = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
                      'otherDebt', 'payable', 'shortDebt', 'longDebt']

    df_lr = df.copy()

    if target_col not in df_lr.columns:
        current_assets = df_lr['cash'] + df_lr['shortReceivable'] + df_lr['inventory'] + df_lr['shortAsset']
        current_liabs = df_lr['payable'] + df_lr['shortDebt'] + df_lr['otherDebt'] + df_lr['longDebt']
        df_lr[target_col] = current_assets / current_liabs
        df_lr[target_col] = df_lr[target_col].replace([np.inf, -np.inf], 0).fillna(0)

    feature_cols = []
    for col in component_cols + [target_col]:
        df_lr[f'{col}_prev'] = df_lr[col].shift(1)
        feature_cols.append(f'{col}_prev')

    df_lr = df_lr.dropna()

    X = df_lr[feature_cols].values
    y = df_lr[target_col].values

    if len(X) < 5:
        return {"name": "Linear Regression", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}

    # Temporal split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    return calculate_metrics(y_test, y_pred, "Multiple Linear Regression")


# ==========================================
# 3. MAIN ENDPOINT
# ==========================================

@router.get("/compare", response_class=HTMLResponse)
async def compare_models(
        request: Request,
        symbol: str = Query('FPT', description="Stock symbol"),
        session: AsyncSession = Depends(session_manager.session)
):
    """
    Runs a benchmark comparison across LSTM, RNN, RF, and MLR
    using REAL Balance Sheet data to predict Current Ratio.
    """
    symbol = symbol.upper()

    # Get available symbols for the UI
    symbols = await get_symbols(session)

    # 1. Fetch Real Data
    try:
        # Fetch data for enough years to run a test
        df_balance = await get_balance_sheet(session, symbol, year=2023, yearly=True)
        df_balance = df_balance.fillna(0)
    except Exception as e:
        return templates.TemplateResponse(
            "benchmark/benchmark_testing.html",
            {
                "request": request,
                "title": f"Model Performance Comparison - Error",
                "models": [],
                "best_model_name": "None",
                "symbols": symbols,
                "symbol": symbol,
                "error": str(e)
            }
        )

    if df_balance.empty:
        return templates.TemplateResponse(
            "benchmark/benchmark_testing.html",
            {
                "request": request,
                "title": f"Model Performance Comparison ({symbol}) - No Data",
                "models": [],
                "best_model_name": "None",
                "symbols": symbols,
                "symbol": symbol
            }
        )

    # 2. Run Benchmarks
    # Target: Current Ratio

    # LSTM (Loads trained_models/current_ratio_{symbol}.keras)
    lstm_metrics = run_dl_model(df_balance, symbol, model_type="LSTM")

    # RNN (Loads trained_models/rnn_current_ratio_{symbol}.keras)
    rnn_metrics = run_dl_model(df_balance, symbol, model_type="RNN")

    # ML Baselines (Predicts Current Ratio directly)
    rf_metrics = run_random_forest(df_balance, target_col="currentRatio")
    lr_metrics = run_linear_regression(df_balance, target_col="currentRatio")

    models_data = [lstm_metrics, rnn_metrics, rf_metrics, lr_metrics]

    # 3. Determine Best Model
    valid_models = [m for m in models_data if isinstance(m['R2'], (int, float))]

    best_model_name = "None"
    if valid_models:
        best_model = max(valid_models, key=lambda x: x['R2'])
        best_model_name = best_model['name']

    # 4. Render
    return templates.TemplateResponse(
        "benchmark/benchmark_testing.html",
        {
            "request": request,
            "title": f"Model Performance Comparison ({symbol} - Current Ratio)",
            "models": models_data,
            "best_model_name": best_model_name,
            "symbols": symbols,
            "symbol": symbol
        }
    )