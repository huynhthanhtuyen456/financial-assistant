# import math
# import numpy as np
# from fastapi import APIRouter, Request, Depends
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
# from sqlalchemy.ext.asyncio import AsyncSession
# from tensorflow.keras.models import load_model
#
# from core.helpers import get_balance_sheet
# from db import session_manager
#
# router = APIRouter(
#     prefix="/models",
#     tags=["models"]
# )
#
# # Setup templates directory (adjust path if needed)
# templates = Jinja2Templates(directory="templates")
#
#
# def calculate_metrics(y_true, y_pred, model_name):
#     """Helper to calculate regression metrics."""
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = math.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#
#     return {
#         "name": model_name,
#         "MSE": round(mse, 4),
#         "RMSE": round(rmse, 4),
#         "MAE": round(mae, 4),
#         "R2": round(r2, 4)
#     }
#
#
# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     data_x, data_y = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         data_x.append(a)
#         data_y.append(dataset[i + look_back, 0])
#     return np.array(data_x), np.array(data_y)
#
#
# def inverse_scale_1d(values, scaler, target_col=0):
#     vals = np.array(values).reshape(-1, 1)
#     n_features = scaler.scale_.shape[0]
#     dummy = np.zeros((len(vals), n_features), dtype=float)
#     dummy[:, target_col] = vals[:, 0]
#     inv = scaler.inverse_transform(dummy)
#     return inv[:, target_col]
#
#
# @router.get("/compare", response_class=HTMLResponse)
# async def compare_models(request: Request, session: AsyncSession = Depends(session_manager.session)):
#     """LSTM Benchmark Testing"""""
#     df_bs = await get_balance_sheet(session, year=2023, yearly=True, symbol="FPT")
#     numeric_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
#     df_bs["currentLiabilities"] = df_bs["payable"] + df_bs["shortDebt"] + df_bs["longDebt"]
#     df_bs["currentAsset"] = df_bs["cash"] + df_bs["shortReceivable"] + \
#                             df_bs["inventory"] + df_bs["shortAsset"] + \
#                             df_bs["otherDebt"]
#     df_bs["currentRatio"] = df_bs["currentAsset"] / df_bs["currentLiabilities"]
#     df_bs["currentRatio"] = df_bs["currentRatio"].replace([np.inf, -np.inf], 0)
#     df_bs["currentRatio"].to_numpy().tolist()
#     df_bs.fillna(0, inplace=True)
#     features_cr = ['currentLiabilities', 'currentRatio', 'currentAsset', 'payable', 'shortDebt', 'longDebt', 'cash',
#                    'shortReceivable', 'inventory', 'shortAsset', 'otherDebt']
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     dataset = df_bs[features_cr].values
#     dataset = dataset.astype('float32')
#     dataset = scaler.fit_transform(dataset)
#     lstm_model = load_model("trained_models/LSTM_BalanceSheet.keras")
#     target_col = features_cr.index('currentRatio')  # index of target column in features_cr
#     # split into train and test sets
#     train_size = int(len(dataset) * 0.6)
#     val_size = int(len(dataset) * 0.8)
#     train, val, test = dataset[0:train_size, :], dataset[train_size:val_size, :], dataset[val_size:, :]
#     # reshape into X=t and Y=t+1
#     look_back = 5
#     trainX, trainY = create_dataset(train, look_back)
#     valX, valY = create_dataset(val, look_back)
#     testX, testY = create_dataset(test, look_back)
#     n_features = len(features_cr)
#     # trainX already has shape (samples, look_back, n_features) from create_dataset_multifeature
#     # Reshape to (samples, look_back, n_features) for LSTM
#     trainX_reshaped = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
#     testX_reshaped = testX.reshape((testX.shape[0], testX.shape[1], 1))
#     valX_reshaped = valX.reshape((valX.shape[0], valX.shape[1], 1))
#     # make predictions
#     trainPredict = lstm_model.predict(trainX_reshaped)
#     testPredict = lstm_model.predict(testX_reshaped)
#
#     # invert predictions and true values
#     trainPredict_inv = inverse_scale_1d(trainPredict, scaler, target_col=target_col)
#     trainY_inv = inverse_scale_1d(trainY, scaler, target_col=target_col)
#     testPredict_inv = inverse_scale_1d(testPredict, scaler, target_col=target_col)
#     testY_inv = inverse_scale_1d(testY, scaler, target_col=target_col)
#
#     # calculate root mean squared error
#     trainScore = np.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
#     testScore = np.sqrt(mean_squared_error(testY_inv, testPredict_inv))
#
#     # shift train predictions for plotting (put values into the target column)
#     trainPredictPlot = np.empty_like(dataset, dtype=float)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[look_back:look_back + len(trainPredict_inv), target_col] = trainPredict_inv
#
#     # shift test predictions for plotting
#     testPredictPlot = np.empty_like(dataset, dtype=float)
#     testPredictPlot[:, :] = np.nan
#     start = len(trainPredict) + (look_back * 2) + 1
#     end = start + len(testPredict_inv)
#     testPredictPlot[start:end, target_col] = testPredict_inv
#     """End LSTM Benchmark Testing"""
#
#     """Random Forest Benchmark Testing"""
#     # Random Forest to predict 'asset' and plot trend for selected symbol (reuses notebook variables)
#
#     # Prepare data: use numeric columns + encoded symbol
#     df = df_bs.copy()
#     le = LabelEncoder()
#     df['symbol_enc'] = le.fit_transform(df['ticker'].astype(str))
#
#     # features: all numeric columns except target 'asset', plus encoded symbol
#     feature_cols = [c for c in features_cr if c != 'currentRatio'] + ['symbol_enc']
#     X = df[features_cr].astype(float)
#     y = df['currentRatio'].astype(float)
#     y.fillna(0, inplace=True)
#
#     # train / test split
#     X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.20, random_state=42)
#
#     # train Random Forest
#     rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
#     rf.fit(X_train_rf, y_train_rf)
#
#     # evaluate
#     y_pred_rf = rf.predict(X_test_rf)
#     """End Random Forest Benchmark Testing"""
#
#     """
#     RNN Prediction Benchmark Testing
#     """
#     trainX_reshaped_rnn = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
#     testX_reshaped_rnn = testX.reshape((testX.shape[0], testX.shape[1], 1))
#     rnn_model = load_model("notebook/notebook/models/RNN_BalanceSheet.keras")
#     trainPred_scaled_rnn = rnn_model.predict(trainX_reshaped_rnn)
#     testPred_scaled_rnn = rnn_model.predict(testX_reshaped_rnn)
#
#     # Inverse scale predictions & true values using existing helper inverse_scale_1d
#     # (expects predictions for target_col in scaled space)
#     trainPred_inv_rnn = inverse_scale_1d(trainPred_scaled_rnn, scaler, target_col=target_col)
#     trainY_inv_rnn = inverse_scale_1d(trainY, scaler, target_col=target_col)
#     testPred_inv_rnn = inverse_scale_1d(testPred_scaled_rnn, scaler, target_col=target_col)
#     testY_inv_rnn = inverse_scale_1d(testY, scaler, target_col=target_col)
#     """
#     End RNN Prediction Benchmark Testing
#     """
#
#     """
#     Multiple Linear Regress Prediction Benchmark Testing
#     """
#     # Linear regression prediction for asset (new cell)
#     sym = "VIC"
#     if sym is None:
#         raise ValueError("No symbol available to predict for.")
#
#     # prepare company history
#     df_sym = df_bs[df_bs['ticker'] == sym].sort_values('year').reset_index(drop=True)
#     if df_sym.empty:
#         raise ValueError(f"No data for symbol {sym}.")
#
#     # prediction target year
#     target_year = 2023
#
#     features_cr = ['currentLiabilities', 'currentRatio', 'currentAsset', 'payable', 'shortDebt', 'longDebt', 'cash',
#                    'shortReceivable', 'inventory', 'shortAsset', 'otherDebt']
#     # features to use (exclude target 'asset')
#     features = features_cr + ["year"]
#     # ensure features exist
#     features = [f for f in features if f in df_sym.columns]
#     if 'year' not in features:
#         raise ValueError("'year' column required in features for regression.")
#
#     # training set: all years strictly before target_year (fallback: use all but last row if none)
#     df_train = df_sym[df_sym['year'] < target_year].copy()
#     if df_train.shape[0] < 2:
#         # fallback to using all but last row to have at least one sample to train/predict
#         if df_sym.shape[0] < 2:
#             raise ValueError(f"Not enough historical rows for symbol {sym} to train linear regression.")
#         df_train = df_sym.iloc[:-1].copy()
#
#     X_train_lr = df_train[features].astype(float).values
#     y_train_lr = df_train['currentRatio'].astype(float).values
#     X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(X_train_lr, y_train_lr, test_size=0.20, random_state=42)
#
#     # fit linear regression
#     lr_model = LinearRegression()
#     lr_model.fit(X_train_lr, y_train_lr)
#
#     # prepare input row for prediction: take last available row before target_year (or last overall), set year=target_year
#     candidate = df_sym[df_sym['year'] < target_year]
#     if candidate.empty:
#         input_row = df_sym.iloc[-1]
#     else:
#         input_row = candidate.iloc[-1]
#
#     X_pred = input_row.copy()
#     X_pred = X_pred[features].astype(float)
#     X_pred['year'] = target_year  # set year to target
#     X_pred_arr = np.array(X_pred).reshape(1, -1)
#
#     pred_asset_lr = float(lr_model.predict(X_pred_arr)[0])
#
#     # evaluate training fit
#     y_pred_mlr = lr_model.predict(X_test_mlr)
#     """
#     Multiple Linear Regression Benchmark Testing
#     """
#     # ---------------------------------------------------------
#     # 1. SIMULATE DATA (Replace this with your actual model outputs)
#     # ---------------------------------------------------------
#     # Let's assume y_test are the actual target values
#
#     # ---------------------------------------------------------
#     # 2. CALCULATE METRICS
#     # ---------------------------------------------------------
#     models_data = [
#         # calculate_metrics(y_test, pred_rnn, "RNN"),
#         calculate_metrics(testY_inv_rnn, testPred_inv_rnn, "RNN"),
#         calculate_metrics(testY_inv, testPredict_inv, "LSTM"),
#         calculate_metrics(y_test_rf, y_pred_rf, "Random Forest"),
#         calculate_metrics(y_test_mlr, y_pred_mlr, "Multiple Linear Regression")
#     ]
#
#     # Find the best model based on R2 Score (closest to 1.0) for highlighting
#     best_model = max(models_data, key=lambda x: x['R2'])
#
#     # ---------------------------------------------------------
#     # 3. RENDER TEMPLATE
#     # ---------------------------------------------------------
#     return templates.TemplateResponse(
#         "benchmark/benchmark_testing.html",
#         {
#             "request": request,
#             "title": "Model Performance Comparison",
#             "models": models_data,
#             "best_model_name": best_model['name']
#         }
#     )


import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ML/DL Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.models import load_model

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


def create_dataset_window(dataset, look_back=1):
    """Converts array into Matrix X and Y for time series."""
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])  # Assuming target is column 0
    return np.array(data_x), np.array(data_y)


def inverse_transform_target(pred_scaled, scaler, n_features, target_col_index=0):
    """
    Inverses scaling for the target column specifically.
    pred_scaled: shape (samples, 1)
    """
    dummy = np.zeros((len(pred_scaled), n_features))
    dummy[:, target_col_index] = pred_scaled.flatten()
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_index]


# ==========================================
# 2. DATA GENERATION
# ==========================================

def generate_dummy_financial_data(n_samples=500):
    """
    Generates a dataframe mimicking balance sheet data.
    """
    np.random.seed(42)

    # Generate random trends with some seasonality/noise
    t = np.linspace(0, 50, n_samples)
    trend = t * 10
    noise = np.random.normal(0, 5, n_samples)

    # Create features
    current_asset = 1000 + trend + np.sin(t) * 100 + noise
    current_liabilities = 500 + (trend * 0.5) + np.cos(t) * 50 + noise

    # Derive other columns to match original code structure
    data = {
        'currentAsset': current_asset,
        'currentLiabilities': current_liabilities,
        'payable': current_liabilities * 0.4,
        'shortDebt': current_liabilities * 0.3,
        'longDebt': current_liabilities * 0.1,
        'cash': current_asset * 0.2,
        'shortReceivable': current_asset * 0.3,
        'inventory': current_asset * 0.3,
        'shortAsset': current_asset * 0.1,
        'otherDebt': current_liabilities * 0.2,
        'ticker': ['FPT'] * n_samples,
        'year': [2020 + (i // 100) for i in range(n_samples)]  # Dummy years
    }

    df = pd.DataFrame(data)

    # Calculate target: Current Ratio
    # Handle division by zero
    df['currentRatio'] = df['currentAsset'] / df['currentLiabilities']
    df['currentRatio'] = df['currentRatio'].replace([np.inf, -np.inf], 0).fillna(0)

    return df


# ==========================================
# 3. MODEL SPECIFIC LOGIC
# ==========================================

async def run_dl_model(df, model_type="LSTM"):
    """
    Generic handler for LSTM and RNN data prep, training, and prediction.
    """
    # Feature selection
    features_cr = ['currentRatio', 'currentLiabilities', 'currentAsset', 'payable',
                   'shortDebt', 'longDebt', 'cash', 'shortReceivable', 'inventory',
                   'shortAsset', 'otherDebt']

    # Data Prep
    dataset = df[features_cr].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Split BEFORE scaling to prevent leakage
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    scaler.fit(train_data)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_data)

    look_back = 5
    X_train, y_train = create_dataset_window(train_scaled, look_back)
    X_test, y_test = create_dataset_window(test_scaled, look_back)

    # Reshape for [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build Model on the fly (Mocking the load_model)
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(32, input_shape=(look_back, len(features_cr))))
        # model = load_model("trained_models/LSTM_BalanceSheet.keras")
    else:
        model.add(SimpleRNN(32, input_shape=(look_back, len(features_cr))))
        # model = load_model("notebook/notebook/models/RNN_BalanceSheet.keras")

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train briefly for benchmark
    model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)

    # Predict
    test_predict_scaled = model.predict(X_test)

    # Invert predictions
    # y_test is 1D array of scaled target values
    # test_predict_scaled is (samples, 1)

    y_test_inv = inverse_transform_target(y_test.reshape(-1, 1), scaler, len(features_cr), 0)
    test_predict_inv = inverse_transform_target(test_predict_scaled, scaler, len(features_cr), 0)

    return calculate_metrics(y_test_inv, test_predict_inv, model_type)


def run_random_forest(df):
    features_cr = ['currentLiabilities', 'currentAsset', 'payable', 'shortDebt',
                   'longDebt', 'cash', 'shortReceivable', 'inventory',
                   'shortAsset', 'otherDebt']

    X = df[features_cr].values
    y = df['currentRatio'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return calculate_metrics(y_test, y_pred, "Random Forest")


def run_linear_regression(df):
    features_cr = ['currentLiabilities', 'currentAsset', 'payable', 'shortDebt',
                   'longDebt', 'cash', 'shortReceivable', 'inventory',
                   'shortAsset', 'otherDebt']

    # Simple temporal split simulation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features_cr].values
    y_train = train_df['currentRatio'].values
    X_test = test_df[features_cr].values
    y_test = test_df['currentRatio'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    return calculate_metrics(y_test, y_pred, "Multiple Linear Regression")


# ==========================================
# 4. MAIN ENDPOINT
# ==========================================

@router.get("/compare", response_class=HTMLResponse)
async def compare_models(request: Request):
    """
    Runs a benchmark comparison across LSTM, RNN, RF, and MLR
    using generated financial data.
    """

    # 1. Generate Data
    df_bs = generate_dummy_financial_data(n_samples=200)

    # 2. Run Benchmarks
    # Note: DL models are async friendly if we wrap them, but here we run sequentially
    lstm_metrics = await run_dl_model(df_bs, model_type="LSTM")
    rnn_metrics = await run_dl_model(df_bs, model_type="RNN")
    rf_metrics = run_random_forest(df_bs)
    lr_metrics = run_linear_regression(df_bs)

    models_data = [lstm_metrics, rnn_metrics, rf_metrics, lr_metrics]

    # 3. Determine Best Model (Highest R2)
    # Filter out potential NaNs or infinite values if training failed
    valid_models = [m for m in models_data if not math.isnan(m['R2'])]

    best_model_name = "None"
    if valid_models:
        best_model = max(valid_models, key=lambda x: x['R2'])
        best_model_name = best_model['name']

    # 4. Render
    return templates.TemplateResponse(
        "benchmark/benchmark_testing.html",
        {
            "request": request,
            "title": "Model Performance Comparison (Benchmark)",
            "models": models_data,
            "best_model_name": best_model_name
        }
    )