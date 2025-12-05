from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sqlalchemy.ext.asyncio import AsyncSession

from core.helpers import get_balance_sheet, get_cash_flow, get_income_statement
import math

from db import session_manager

router = APIRouter(
    prefix="/models",
    tags=["models"]
)

# Setup templates directory (adjust path if needed)
templates = Jinja2Templates(directory="templates")


def calculate_metrics(y_true, y_pred, model_name):
    """Helper to calculate regression metrics."""
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


@router.get("/compare", response_class=HTMLResponse)
async def compare_models(request: Request, session: AsyncSession = Depends(session_manager.session)):
    # ---------------------------------------------------------
    # 1. SIMULATE DATA (Replace this with your actual model outputs)
    # ---------------------------------------------------------
    # Let's assume y_test are the actual target values
    df_bs = await get_balance_sheet(session, year=2023, yearly=True, symbol="FPT")
    numeric_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
    df_bs["currentLiabilities"] = df_bs["payable"] + df_bs["shortDebt"] + df_bs["longDebt"]
    df_bs["currentAsset"] = df_bs["cash"] + df_bs["shortReceivable"] + \
                                       df_bs["inventory"] + df_bs["shortAsset"] + \
                                       df_bs["otherDebt"]
    df_bs["currentRatio"] = df_bs["currentAsset"] / df_bs["currentLiabilities"]
    df_bs["currentRatio"] = df_bs["currentRatio"].replace([np.inf, -np.inf], 0)
    df_bs["currentRatio"].to_numpy().tolist()
    df_bs.fillna(0, inplace=True)

    y_test = df_bs["currentRatio"].to_numpy()
    y_test = np.random.rand(100) * 100
    # Simulate predictions for each model with varying noise/accuracy
    # RNN (Assume slightly noisy)
    pred_rnn = y_test + np.random.normal(0, 10, 100)
    # LSTM (Assume better time-series handling, less noise)
    pred_lstm = y_test + np.random.normal(0, 5, 100)
    # Random Forest (Good non-linear fit)
    pred_rf = y_test + np.random.normal(0, 8, 100)
    # Multiple Linear Regression (Linear fit, might underfit complex data)
    pred_mlr = y_test + np.random.normal(0, 12, 100)

    # ---------------------------------------------------------
    # 2. CALCULATE METRICS
    # ---------------------------------------------------------
    models_data = [
        calculate_metrics(y_test, pred_rnn, "RNN"),
        calculate_metrics(y_test, pred_lstm, "LSTM"),
        calculate_metrics(y_test, pred_rf, "Random Forest"),
        calculate_metrics(y_test, pred_mlr, "Multiple Linear Regression")
    ]

    # Find the best model based on R2 Score (closest to 1.0) for highlighting
    best_model = max(models_data, key=lambda x: x['R2'])

    # ---------------------------------------------------------
    # 3. RENDER TEMPLATE
    # ---------------------------------------------------------
    return templates.TemplateResponse(
        "benchmark/benchmark_testing.html",
        {
            "request": request,
            "title": "Model Performance Comparison",
            "models": models_data,
            "best_model_name": best_model['name']
        }
    )