# # # # import math
# # # # import numpy as np
# # # # from fastapi import APIRouter, Request, Depends
# # # # from fastapi.responses import HTMLResponse
# # # # from fastapi.templating import Jinja2Templates
# # # # from sklearn.ensemble import RandomForestRegressor
# # # # from sklearn.linear_model import LinearRegression
# # # # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.preprocessing import LabelEncoder
# # # # from sklearn.preprocessing import MinMaxScaler
# # # # from sqlalchemy.ext.asyncio import AsyncSession
# # # # from tensorflow.keras.models import load_model
# # # #
# # # # from core.helpers import get_balance_sheet
# # # # from db import session_manager
# # # #
# # # # router = APIRouter(
# # # #     prefix="/models",
# # # #     tags=["models"]
# # # # )
# # # #
# # # # # Setup templates directory (adjust path if needed)
# # # # templates = Jinja2Templates(directory="templates")
# # # #
# # # #
# # # # def calculate_metrics(y_true, y_pred, model_name):
# # # #     """Helper to calculate regression metrics."""
# # # #     mse = mean_squared_error(y_true, y_pred)
# # # #     rmse = math.sqrt(mse)
# # # #     mae = mean_absolute_error(y_true, y_pred)
# # # #     r2 = r2_score(y_true, y_pred)
# # # #
# # # #     return {
# # # #         "name": model_name,
# # # #         "MSE": round(mse, 4),
# # # #         "RMSE": round(rmse, 4),
# # # #         "MAE": round(mae, 4),
# # # #         "R2": round(r2, 4)
# # # #     }
# # # #
# # # #
# # # # # convert an array of values into a dataset matrix
# # # # def create_dataset(dataset, look_back=1):
# # # #     data_x, data_y = [], []
# # # #     for i in range(len(dataset)-look_back-1):
# # # #         a = dataset[i:(i+look_back), 0]
# # # #         data_x.append(a)
# # # #         data_y.append(dataset[i + look_back, 0])
# # # #     return np.array(data_x), np.array(data_y)
# # # #
# # # #
# # # # def inverse_scale_1d(values, scaler, target_col=0):
# # # #     vals = np.array(values).reshape(-1, 1)
# # # #     n_features = scaler.scale_.shape[0]
# # # #     dummy = np.zeros((len(vals), n_features), dtype=float)
# # # #     dummy[:, target_col] = vals[:, 0]
# # # #     inv = scaler.inverse_transform(dummy)
# # # #     return inv[:, target_col]
# # # #
# # # #
# # # # @router.get("/compare", response_class=HTMLResponse)
# # # # async def compare_models(request: Request, session: AsyncSession = Depends(session_manager.session)):
# # # #     """LSTM Benchmark Testing"""""
# # # #     df_bs = await get_balance_sheet(session, year=2023, yearly=True, symbol="FPT")
# # # #     numeric_cols = df_bs.select_dtypes(include=[np.number]).columns.tolist()
# # # #     df_bs["currentLiabilities"] = df_bs["payable"] + df_bs["shortDebt"] + df_bs["longDebt"]
# # # #     df_bs["currentAsset"] = df_bs["cash"] + df_bs["shortReceivable"] + \
# # # #                             df_bs["inventory"] + df_bs["shortAsset"] + \
# # # #                             df_bs["otherDebt"]
# # # #     df_bs["currentRatio"] = df_bs["currentAsset"] / df_bs["currentLiabilities"]
# # # #     df_bs["currentRatio"] = df_bs["currentRatio"].replace([np.inf, -np.inf], 0)
# # # #     df_bs["currentRatio"].to_numpy().tolist()
# # # #     df_bs.fillna(0, inplace=True)
# # # #     features_cr = ['currentLiabilities', 'currentRatio', 'currentAsset', 'payable', 'shortDebt', 'longDebt', 'cash',
# # # #                    'shortReceivable', 'inventory', 'shortAsset', 'otherDebt']
# # # #     scaler = MinMaxScaler(feature_range=(0, 1))
# # # #     dataset = df_bs[features_cr].values
# # # #     dataset = dataset.astype('float32')
# # # #     dataset = scaler.fit_transform(dataset)
# # # #     lstm_model = load_model("trained_models/LSTM_BalanceSheet.keras")
# # # #     target_col = features_cr.index('currentRatio')  # index of target column in features_cr
# # # #     # split into train and test sets
# # # #     train_size = int(len(dataset) * 0.6)
# # # #     val_size = int(len(dataset) * 0.8)
# # # #     train, val, test = dataset[0:train_size, :], dataset[train_size:val_size, :], dataset[val_size:, :]
# # # #     # reshape into X=t and Y=t+1
# # # #     look_back = 5
# # # #     trainX, trainY = create_dataset(train, look_back)
# # # #     valX, valY = create_dataset(val, look_back)
# # # #     testX, testY = create_dataset(test, look_back)
# # # #     n_features = len(features_cr)
# # # #     # trainX already has shape (samples, look_back, n_features) from create_dataset_multifeature
# # # #     # Reshape to (samples, look_back, n_features) for LSTM
# # # #     trainX_reshaped = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
# # # #     testX_reshaped = testX.reshape((testX.shape[0], testX.shape[1], 1))
# # # #     valX_reshaped = valX.reshape((valX.shape[0], valX.shape[1], 1))
# # # #     # make predictions
# # # #     trainPredict = lstm_model.predict(trainX_reshaped)
# # # #     testPredict = lstm_model.predict(testX_reshaped)
# # # #
# # # #     # invert predictions and true values
# # # #     trainPredict_inv = inverse_scale_1d(trainPredict, scaler, target_col=target_col)
# # # #     trainY_inv = inverse_scale_1d(trainY, scaler, target_col=target_col)
# # # #     testPredict_inv = inverse_scale_1d(testPredict, scaler, target_col=target_col)
# # # #     testY_inv = inverse_scale_1d(testY, scaler, target_col=target_col)
# # # #
# # # #     # calculate root mean squared error
# # # #     trainScore = np.sqrt(mean_squared_error(trainY_inv, trainPredict_inv))
# # # #     testScore = np.sqrt(mean_squared_error(testY_inv, testPredict_inv))
# # # #
# # # #     # shift train predictions for plotting (put values into the target column)
# # # #     trainPredictPlot = np.empty_like(dataset, dtype=float)
# # # #     trainPredictPlot[:, :] = np.nan
# # # #     trainPredictPlot[look_back:look_back + len(trainPredict_inv), target_col] = trainPredict_inv
# # # #
# # # #     # shift test predictions for plotting
# # # #     testPredictPlot = np.empty_like(dataset, dtype=float)
# # # #     testPredictPlot[:, :] = np.nan
# # # #     start = len(trainPredict) + (look_back * 2) + 1
# # # #     end = start + len(testPredict_inv)
# # # #     testPredictPlot[start:end, target_col] = testPredict_inv
# # # #     """End LSTM Benchmark Testing"""
# # # #
# # # #     """Random Forest Benchmark Testing"""
# # # #     # Random Forest to predict 'asset' and plot trend for selected symbol (reuses notebook variables)
# # # #
# # # #     # Prepare data: use numeric columns + encoded symbol
# # # #     df = df_bs.copy()
# # # #     le = LabelEncoder()
# # # #     df['symbol_enc'] = le.fit_transform(df['ticker'].astype(str))
# # # #
# # # #     # features: all numeric columns except target 'asset', plus encoded symbol
# # # #     feature_cols = [c for c in features_cr if c != 'currentRatio'] + ['symbol_enc']
# # # #     X = df[features_cr].astype(float)
# # # #     y = df['currentRatio'].astype(float)
# # # #     y.fillna(0, inplace=True)
# # # #
# # # #     # train / test split
# # # #     X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.20, random_state=42)
# # # #
# # # #     # train Random Forest
# # # #     rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
# # # #     rf.fit(X_train_rf, y_train_rf)
# # # #
# # # #     # evaluate
# # # #     y_pred_rf = rf.predict(X_test_rf)
# # # #     """End Random Forest Benchmark Testing"""
# # # #
# # # #     """
# # # #     RNN Prediction Benchmark Testing
# # # #     """
# # # #     trainX_reshaped_rnn = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
# # # #     testX_reshaped_rnn = testX.reshape((testX.shape[0], testX.shape[1], 1))
# # # #     rnn_model = load_model("notebook/notebook/models/RNN_BalanceSheet.keras")
# # # #     trainPred_scaled_rnn = rnn_model.predict(trainX_reshaped_rnn)
# # # #     testPred_scaled_rnn = rnn_model.predict(testX_reshaped_rnn)
# # # #
# # # #     # Inverse scale predictions & true values using existing helper inverse_scale_1d
# # # #     # (expects predictions for target_col in scaled space)
# # # #     trainPred_inv_rnn = inverse_scale_1d(trainPred_scaled_rnn, scaler, target_col=target_col)
# # # #     trainY_inv_rnn = inverse_scale_1d(trainY, scaler, target_col=target_col)
# # # #     testPred_inv_rnn = inverse_scale_1d(testPred_scaled_rnn, scaler, target_col=target_col)
# # # #     testY_inv_rnn = inverse_scale_1d(testY, scaler, target_col=target_col)
# # # #     """
# # # #     End RNN Prediction Benchmark Testing
# # # #     """
# # # #
# # # #     """
# # # #     Multiple Linear Regress Prediction Benchmark Testing
# # # #     """
# # # #     # Linear regression prediction for asset (new cell)
# # # #     sym = "VIC"
# # # #     if sym is None:
# # # #         raise ValueError("No symbol available to predict for.")
# # # #
# # # #     # prepare company history
# # # #     df_sym = df_bs[df_bs['ticker'] == sym].sort_values('year').reset_index(drop=True)
# # # #     if df_sym.empty:
# # # #         raise ValueError(f"No data for symbol {sym}.")
# # # #
# # # #     # prediction target year
# # # #     target_year = 2023
# # # #
# # # #     features_cr = ['currentLiabilities', 'currentRatio', 'currentAsset', 'payable', 'shortDebt', 'longDebt', 'cash',
# # # #                    'shortReceivable', 'inventory', 'shortAsset', 'otherDebt']
# # # #     # features to use (exclude target 'asset')
# # # #     features = features_cr + ["year"]
# # # #     # ensure features exist
# # # #     features = [f for f in features if f in df_sym.columns]
# # # #     if 'year' not in features:
# # # #         raise ValueError("'year' column required in features for regression.")
# # # #
# # # #     # training set: all years strictly before target_year (fallback: use all but last row if none)
# # # #     df_train = df_sym[df_sym['year'] < target_year].copy()
# # # #     if df_train.shape[0] < 2:
# # # #         # fallback to using all but last row to have at least one sample to train/predict
# # # #         if df_sym.shape[0] < 2:
# # # #             raise ValueError(f"Not enough historical rows for symbol {sym} to train linear regression.")
# # # #         df_train = df_sym.iloc[:-1].copy()
# # # #
# # # #     X_train_lr = df_train[features].astype(float).values
# # # #     y_train_lr = df_train['currentRatio'].astype(float).values
# # # #     X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(X_train_lr, y_train_lr, test_size=0.20, random_state=42)
# # # #
# # # #     # fit linear regression
# # # #     lr_model = LinearRegression()
# # # #     lr_model.fit(X_train_lr, y_train_lr)
# # # #
# # # #     # prepare input row for prediction: take last available row before target_year (or last overall), set year=target_year
# # # #     candidate = df_sym[df_sym['year'] < target_year]
# # # #     if candidate.empty:
# # # #         input_row = df_sym.iloc[-1]
# # # #     else:
# # # #         input_row = candidate.iloc[-1]
# # # #
# # # #     X_pred = input_row.copy()
# # # #     X_pred = X_pred[features].astype(float)
# # # #     X_pred['year'] = target_year  # set year to target
# # # #     X_pred_arr = np.array(X_pred).reshape(1, -1)
# # # #
# # # #     pred_asset_lr = float(lr_model.predict(X_pred_arr)[0])
# # # #
# # # #     # evaluate training fit
# # # #     y_pred_mlr = lr_model.predict(X_test_mlr)
# # # #     """
# # # #     Multiple Linear Regression Benchmark Testing
# # # #     """
# # # #     # ---------------------------------------------------------
# # # #     # 1. SIMULATE DATA (Replace this with your actual model outputs)
# # # #     # ---------------------------------------------------------
# # # #     # Let's assume y_test are the actual target values
# # # #
# # # #     # ---------------------------------------------------------
# # # #     # 2. CALCULATE METRICS
# # # #     # ---------------------------------------------------------
# # # #     models_data = [
# # # #         # calculate_metrics(y_test, pred_rnn, "RNN"),
# # # #         calculate_metrics(testY_inv_rnn, testPred_inv_rnn, "RNN"),
# # # #         calculate_metrics(testY_inv, testPredict_inv, "LSTM"),
# # # #         calculate_metrics(y_test_rf, y_pred_rf, "Random Forest"),
# # # #         calculate_metrics(y_test_mlr, y_pred_mlr, "Multiple Linear Regression")
# # # #     ]
# # # #
# # # #     # Find the best model based on R2 Score (closest to 1.0) for highlighting
# # # #     best_model = max(models_data, key=lambda x: x['R2'])
# # # #
# # # #     # ---------------------------------------------------------
# # # #     # 3. RENDER TEMPLATE
# # # #     # ---------------------------------------------------------
# # # #     return templates.TemplateResponse(
# # # #         "benchmark/benchmark_testing.html",
# # # #         {
# # # #             "request": request,
# # # #             "title": "Model Performance Comparison",
# # # #             "models": models_data,
# # # #             "best_model_name": best_model['name']
# # # #         }
# # # #     )
# # #
# # #
# # # import math
# # # import numpy as np
# # # import pandas as pd
# # # from fastapi import APIRouter, Request
# # # from fastapi.responses import HTMLResponse
# # # from fastapi.templating import Jinja2Templates
# # #
# # # # ML/DL Libraries
# # # from sklearn.ensemble import RandomForestRegressor
# # # from sklearn.linear_model import LinearRegression
# # # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.preprocessing import MinMaxScaler
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
# # # from tensorflow.keras.models import load_model
# # #
# # # router = APIRouter(
# # #     prefix="/models",
# # #     tags=["models"]
# # # )
# # #
# # # templates = Jinja2Templates(directory="templates")
# # #
# # #
# # # # ==========================================
# # # # 1. HELPER FUNCTIONS & METRICS
# # # # ==========================================
# # #
# # # def calculate_metrics(y_true, y_pred, model_name):
# # #     """Helper to calculate regression metrics."""
# # #     # Ensure inputs are 1D arrays
# # #     y_true = np.array(y_true).flatten()
# # #     y_pred = np.array(y_pred).flatten()
# # #
# # #     mse = mean_squared_error(y_true, y_pred)
# # #     rmse = math.sqrt(mse)
# # #     mae = mean_absolute_error(y_true, y_pred)
# # #     r2 = r2_score(y_true, y_pred)
# # #
# # #     return {
# # #         "name": model_name,
# # #         "MSE": round(mse, 4),
# # #         "RMSE": round(rmse, 4),
# # #         "MAE": round(mae, 4),
# # #         "R2": round(r2, 4)
# # #     }
# # #
# # #
# # # def create_dataset_window(dataset, look_back=1):
# # #     """Converts array into Matrix X and Y for time series."""
# # #     data_x, data_y = [], []
# # #     for i in range(len(dataset) - look_back - 1):
# # #         a = dataset[i:(i + look_back), :]
# # #         data_x.append(a)
# # #         data_y.append(dataset[i + look_back, 0])  # Assuming target is column 0
# # #     return np.array(data_x), np.array(data_y)
# # #
# # #
# # # def inverse_transform_target(pred_scaled, scaler, n_features, target_col_index=0):
# # #     """
# # #     Inverses scaling for the target column specifically.
# # #     pred_scaled: shape (samples, 1)
# # #     """
# # #     dummy = np.zeros((len(pred_scaled), n_features))
# # #     dummy[:, target_col_index] = pred_scaled.flatten()
# # #     inv = scaler.inverse_transform(dummy)
# # #     return inv[:, target_col_index]
# # #
# # #
# # # # ==========================================
# # # # 2. DATA GENERATION
# # # # ==========================================
# # #
# # # def generate_dummy_financial_data(n_samples=500):
# # #     """
# # #     Generates a dataframe mimicking balance sheet data.
# # #     """
# # #     np.random.seed(42)
# # #
# # #     # Generate random trends with some seasonality/noise
# # #     t = np.linspace(0, 50, n_samples)
# # #     trend = t * 10
# # #     noise = np.random.normal(0, 5, n_samples)
# # #
# # #     # Create features
# # #     current_asset = 1000 + trend + np.sin(t) * 100 + noise
# # #     current_liabilities = 500 + (trend * 0.5) + np.cos(t) * 50 + noise
# # #
# # #     # Derive other columns to match original code structure
# # #     data = {
# # #         'currentAsset': current_asset,
# # #         'currentLiabilities': current_liabilities,
# # #         'payable': current_liabilities * 0.4,
# # #         'shortDebt': current_liabilities * 0.3,
# # #         'longDebt': current_liabilities * 0.1,
# # #         'cash': current_asset * 0.2,
# # #         'shortReceivable': current_asset * 0.3,
# # #         'inventory': current_asset * 0.3,
# # #         'shortAsset': current_asset * 0.1,
# # #         'otherDebt': current_liabilities * 0.2,
# # #         'ticker': ['FPT'] * n_samples,
# # #         'year': [2020 + (i // 100) for i in range(n_samples)]  # Dummy years
# # #     }
# # #
# # #     df = pd.DataFrame(data)
# # #
# # #     # Calculate target: Current Ratio
# # #     # Handle division by zero
# # #     df['currentRatio'] = df['currentAsset'] / df['currentLiabilities']
# # #     df['currentRatio'] = df['currentRatio'].replace([np.inf, -np.inf], 0).fillna(0)
# # #
# # #     return df
# # #
# # #
# # # # ==========================================
# # # # 3. MODEL SPECIFIC LOGIC
# # # # ==========================================
# # #
# # # async def run_dl_model(df, model_type="LSTM"):
# # #     """
# # #     Generic handler for LSTM and RNN data prep, training, and prediction.
# # #     """
# # #     # Feature selection
# # #     features_cr = ['currentRatio', 'currentLiabilities', 'currentAsset', 'payable',
# # #                    'shortDebt', 'longDebt', 'cash', 'shortReceivable', 'inventory',
# # #                    'shortAsset', 'otherDebt']
# # #
# # #     # Data Prep
# # #     dataset = df[features_cr].values.astype('float32')
# # #     scaler = MinMaxScaler(feature_range=(0, 1))
# # #
# # #     # Split BEFORE scaling to prevent leakage
# # #     train_size = int(len(dataset) * 0.8)
# # #     train_data = dataset[:train_size]
# # #     test_data = dataset[train_size:]
# # #
# # #     scaler.fit(train_data)
# # #     train_scaled = scaler.transform(train_data)
# # #     test_scaled = scaler.transform(test_data)
# # #
# # #     look_back = 5
# # #     X_train, y_train = create_dataset_window(train_scaled, look_back)
# # #     X_test, y_test = create_dataset_window(test_scaled, look_back)
# # #
# # #     # Reshape for [samples, time steps, features]
# # #     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
# # #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
# # #
# # #     # Build Model on the fly (Mocking the load_model)
# # #     model = Sequential()
# # #     if model_type == "LSTM":
# # #         model.add(LSTM(32, input_shape=(look_back, len(features_cr))))
# # #         # model = load_model("trained_models/LSTM_BalanceSheet.keras")
# # #     else:
# # #         model.add(SimpleRNN(32, input_shape=(look_back, len(features_cr))))
# # #         # model = load_model("notebook/notebook/models/RNN_BalanceSheet.keras")
# # #
# # #     model.add(Dense(1))
# # #     model.compile(loss='mean_squared_error', optimizer='adam')
# # #
# # #     # Train briefly for benchmark
# # #     model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)
# # #
# # #     # Predict
# # #     test_predict_scaled = model.predict(X_test)
# # #
# # #     # Invert predictions
# # #     # y_test is 1D array of scaled target values
# # #     # test_predict_scaled is (samples, 1)
# # #
# # #     y_test_inv = inverse_transform_target(y_test.reshape(-1, 1), scaler, len(features_cr), 0)
# # #     test_predict_inv = inverse_transform_target(test_predict_scaled, scaler, len(features_cr), 0)
# # #
# # #     return calculate_metrics(y_test_inv, test_predict_inv, model_type)
# # #
# # #
# # # def run_random_forest(df):
# # #     features_cr = ['currentLiabilities', 'currentAsset', 'payable', 'shortDebt',
# # #                    'longDebt', 'cash', 'shortReceivable', 'inventory',
# # #                    'shortAsset', 'otherDebt']
# # #
# # #     X = df[features_cr].values
# # #     y = df['currentRatio'].values
# # #
# # #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # #
# # #     rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
# # #     rf.fit(X_train, y_train)
# # #     y_pred = rf.predict(X_test)
# # #
# # #     return calculate_metrics(y_test, y_pred, "Random Forest")
# # #
# # #
# # # def run_linear_regression(df):
# # #     features_cr = ['currentLiabilities', 'currentAsset', 'payable', 'shortDebt',
# # #                    'longDebt', 'cash', 'shortReceivable', 'inventory',
# # #                    'shortAsset', 'otherDebt']
# # #
# # #     # Simple temporal split simulation
# # #     split_idx = int(len(df) * 0.8)
# # #     train_df = df.iloc[:split_idx]
# # #     test_df = df.iloc[split_idx:]
# # #
# # #     X_train = train_df[features_cr].values
# # #     y_train = train_df['currentRatio'].values
# # #     X_test = test_df[features_cr].values
# # #     y_test = test_df['currentRatio'].values
# # #
# # #     lr = LinearRegression()
# # #     lr.fit(X_train, y_train)
# # #     y_pred = lr.predict(X_test)
# # #
# # #     return calculate_metrics(y_test, y_pred, "Multiple Linear Regression")
# # #
# # #
# # # # ==========================================
# # # # 4. MAIN ENDPOINT
# # # # ==========================================
# # #
# # # @router.get("/compare", response_class=HTMLResponse)
# # # async def compare_models(request: Request):
# # #     """
# # #     Runs a benchmark comparison across LSTM, RNN, RF, and MLR
# # #     using generated financial data.
# # #     """
# # #
# # #     # 1. Generate Data
# # #     df_bs = generate_dummy_financial_data(n_samples=200)
# # #
# # #     # 2. Run Benchmarks
# # #     # Note: DL models are async friendly if we wrap them, but here we run sequentially
# # #     lstm_metrics = await run_dl_model(df_bs, model_type="LSTM")
# # #     rnn_metrics = await run_dl_model(df_bs, model_type="RNN")
# # #     rf_metrics = run_random_forest(df_bs)
# # #     lr_metrics = run_linear_regression(df_bs)
# # #
# # #     models_data = [lstm_metrics, rnn_metrics, rf_metrics, lr_metrics]
# # #
# # #     # 3. Determine Best Model (Highest R2)
# # #     # Filter out potential NaNs or infinite values if training failed
# # #     valid_models = [m for m in models_data if not math.isnan(m['R2'])]
# # #
# # #     best_model_name = "None"
# # #     if valid_models:
# # #         best_model = max(valid_models, key=lambda x: x['R2'])
# # #         best_model_name = best_model['name']
# # #
# # #     # 4. Render
# # #     return templates.TemplateResponse(
# # #         "benchmark/benchmark_testing.html",
# # #         {
# # #             "request": request,
# # #             "title": "Model Performance Comparison (Benchmark)",
# # #             "models": models_data,
# # #             "best_model_name": best_model_name
# # #         }
# # #     )
# #
# #
# # import math
# # import numpy as np
# # import pandas as pd
# # import os
# # from fastapi import APIRouter, Request, Depends
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# #
# # # ML/DL Libraries
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import MinMaxScaler
# # from tensorflow.keras.models import load_model
# # from sqlalchemy.ext.asyncio import AsyncSession
# #
# # # Project Modules
# # from core.helpers import get_income_statement, create_sequences
# # from db import session_manager
# #
# # router = APIRouter(
# #     prefix="/models",
# #     tags=["models"]
# # )
# #
# # templates = Jinja2Templates(directory="templates")
# #
# #
# # # ==========================================
# # # 1. HELPER FUNCTIONS & METRICS
# # # ==========================================
# #
# # def calculate_metrics(y_true, y_pred, model_name):
# #     """Helper to calculate regression metrics."""
# #     # Ensure inputs are 1D arrays
# #     y_true = np.array(y_true).flatten()
# #     y_pred = np.array(y_pred).flatten()
# #
# #     # Handle edge case where predictions are empty or NaN
# #     if len(y_true) == 0 or len(y_pred) == 0:
# #         return {
# #             "name": model_name,
# #             "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0
# #         }
# #
# #     mse = mean_squared_error(y_true, y_pred)
# #     rmse = math.sqrt(mse)
# #     mae = mean_absolute_error(y_true, y_pred)
# #     r2 = r2_score(y_true, y_pred)
# #
# #     return {
# #         "name": model_name,
# #         "MSE": round(mse, 4),
# #         "RMSE": round(rmse, 4),
# #         "MAE": round(mae, 4),
# #         "R2": round(r2, 4)
# #     }
# #
# #
# # def inverse_transform_target(pred_scaled, scaler, n_features, target_col_index):
# #     """
# #     Inverses scaling for the target column specifically.
# #     pred_scaled: shape (samples, 1) or (samples,)
# #     """
# #     pred_scaled = pred_scaled.flatten()
# #     dummy = np.zeros((len(pred_scaled), n_features))
# #     dummy[:, target_col_index] = pred_scaled
# #     inv = scaler.inverse_transform(dummy)
# #     return inv[:, target_col_index]
# #
# #
# # # ==========================================
# # # 2. MODEL SPECIFIC LOGIC
# # # ==========================================
# #
# # def run_dl_model(df, symbol, model_type="LSTM", target_col="net_profit_margin"):
# #     """
# #     Loads saved LSTM/RNN models and performs prediction on the test set.
# #     """
# #     # Features used in income_statement.py
# #     features = ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']
# #
# #     # Ensure data is sorted
# #     df = df.sort_values('year')
# #     df = df.dropna(subset=features).reset_index(drop=True)
# #
# #     if len(df) < 5:
# #         return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     # Data Prep
# #     dataset = df[features].values.astype('float32')
# #     scaler = MinMaxScaler(feature_range=(0, 1))
# #
# #     # We must fit the scaler on the whole dataset or training part exactly as the dashboard does.
# #     # The dashboard fits on the whole provided history to plot trends,
# #     # but for strict benchmarking we usually split.
# #     # To match 'saved models' input distribution, we fit on the data available.
# #     scaler.fit(dataset)
# #     scaled_data = scaler.transform(dataset)
# #
# #     # Sequence Creation
# #     look_back = 3
# #     X, y = create_sequences(scaled_data, look_back)
# #
# #     if len(X) == 0:
# #         return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     # Split into Train/Test (Last 20% as test)
# #     split_idx = int(len(X) * 0.8)
# #     X_test = X[split_idx:]
# #     y_test = y[split_idx:]
# #
# #     # Load Model
# #     if model_type == "LSTM":
# #         model_path = f"models/income_statement_{symbol}.keras"
# #     else:
# #         model_path = f"models/rnn_income_statement_{symbol}.keras"
# #
# #     if not os.path.exists(model_path):
# #         return {"name": f"{model_type} (Not Found)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     try:
# #         model = load_model(model_path)
# #     except Exception as e:
# #         return {"name": f"{model_type} (Error)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     # Predict
# #     # X_test shape from create_sequences is (samples, look_back, features)
# #     # The saved models expect (samples, look_back, features)
# #     test_predict_scaled = model.predict(X_test, verbose=0)
# #
# #     # Invert predictions
# #     # y contains the vector of features at t+1.
# #     # We need to calculate the margin from the inverted features to compare with actual margin?
# #     # Or does the benchmark compare the raw features?
# #     # For simplicity, let's calculate R2 on the 'postTaxProfit' (index 3) which drives Net Profit Margin.
# #
# #     # Indices in features list: 'grossProfit'(0), 'revenue'(1), 'preTaxProfit'(2), 'postTaxProfit'(3)
# #     target_feat_idx = 3  # postTaxProfit
# #
# #     y_test_inv = inverse_transform_target(y_test[:, target_feat_idx], scaler, len(features), target_feat_idx)
# #     test_predict_inv = inverse_transform_target(test_predict_scaled[:, target_feat_idx], scaler, len(features),
# #                                                 target_feat_idx)
# #
# #     # If we want to benchmark the Margin % itself, we need Revenue (index 1) as well.
# #     # For this benchmark, let's strictly compare the Model's ability to predict Post Tax Profit.
# #     return calculate_metrics(y_test_inv, test_predict_inv, f"{model_type} (Post Tax Profit)")
# #
# #
# # def run_random_forest(df, target_col="postTaxProfit"):
# #     # Features used for regression
# #     features = ['grossProfit', 'revenue', 'preTaxProfit']  # Predict postTax based on others + history?
# #     # Simple lag features for RF to mimic time series nature
# #     df_rf = df.copy()
# #     for col in ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']:
# #         df_rf[f'{col}_prev'] = df_rf[col].shift(1)
# #
# #     df_rf = df_rf.dropna()
# #
# #     feature_cols = [f'{c}_prev' for c in ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']]
# #
# #     X = df_rf[feature_cols].values
# #     y = df_rf[target_col].values
# #
# #     if len(X) < 5:
# #         return {"name": "Random Forest", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# #
# #     rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
# #     rf.fit(X_train, y_train)
# #     y_pred = rf.predict(X_test)
# #
# #     return calculate_metrics(y_test, y_pred, "Random Forest")
# #
# #
# # def run_linear_regression(df, target_col="postTaxProfit"):
# #     # Simple lag features
# #     df_lr = df.copy()
# #     for col in ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']:
# #         df_lr[f'{col}_prev'] = df_lr[col].shift(1)
# #
# #     df_lr = df_lr.dropna()
# #
# #     feature_cols = [f'{c}_prev' for c in ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']]
# #
# #     X = df_lr[feature_cols].values
# #     y = df_lr[target_col].values
# #
# #     if len(X) < 5:
# #         return {"name": "Linear Regression", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
# #
# #     # Temporal split
# #     split_idx = int(len(X) * 0.8)
# #     X_train, X_test = X[:split_idx], X[split_idx:]
# #     y_train, y_test = y[:split_idx], y[split_idx:]
# #
# #     lr = LinearRegression()
# #     lr.fit(X_train, y_train)
# #     y_pred = lr.predict(X_test)
# #
# #     return calculate_metrics(y_test, y_pred, "Multiple Linear Regression")
# #
# #
# # # ==========================================
# # # 3. MAIN ENDPOINT
# # # ==========================================
# #
# # @router.get("/compare", response_class=HTMLResponse)
# # async def compare_models(request: Request, session: AsyncSession = Depends(session_manager.session)):
# #     """
# #     Runs a benchmark comparison across LSTM, RNN, RF, and MLR
# #     using REAL Income Statement data and SAVED models.
# #     """
# #     symbol = "FPT"  # Default for benchmark
# #
# #     # 1. Fetch Real Data
# #     try:
# #         # Fetch data for enough years to run a test
# #         df_income = await get_income_statement(session, symbol, year=2023, yearly=True)
# #         # Calculate margins if needed, though models predict raw values
# #         df_income = df_income.fillna(0)
# #     except Exception as e:
# #         return HTMLResponse(f"Error fetching data: {str(e)}")
# #
# #     if df_income.empty:
# #         return HTMLResponse("No data available for benchmarking.")
# #
# #     # 2. Run Benchmarks
# #     # Note: We compare prediction of "Post Tax Profit" as a proxy for performance
# #
# #     # LSTM (Loads models/income_statement_FPT.keras)
# #     lstm_metrics = run_dl_model(df_income, symbol, model_type="LSTM")
# #
# #     # RNN (Loads models/rnn_income_statement_FPT.keras)
# #     rnn_metrics = run_dl_model(df_income, symbol, model_type="RNN")
# #
# #     # ML Baselines
# #     rf_metrics = run_random_forest(df_income, target_col="postTaxProfit")
# #     lr_metrics = run_linear_regression(df_income, target_col="postTaxProfit")
# #
# #     models_data = [lstm_metrics, rnn_metrics, rf_metrics, lr_metrics]
# #
# #     # 3. Determine Best Model
# #     valid_models = [m for m in models_data if isinstance(m['R2'], (int, float))]
# #
# #     best_model_name = "None"
# #     if valid_models:
# #         best_model = max(valid_models, key=lambda x: x['R2'])
# #         best_model_name = best_model['name']
# #
# #     # 4. Render
# #     return templates.TemplateResponse(
# #         "benchmark/benchmark_testing.html",
# #         {
# #             "request": request,
# #             "title": f"Model Performance Comparison ({symbol} - Post Tax Profit)",
# #             "models": models_data,
# #             "best_model_name": best_model_name
# #         }
# #     )
#
#
# import math
# import numpy as np
# import pandas as pd
# import os
# from fastapi import APIRouter, Request, Depends
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
#
# # ML/DL Libraries
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model
# from sqlalchemy.ext.asyncio import AsyncSession
#
# # Project Modules
# from core.helpers import get_balance_sheet, create_sequences
# from db import session_manager
#
# router = APIRouter(
#     prefix="/models",
#     tags=["models"]
# )
#
# templates = Jinja2Templates(directory="templates")
#
#
# # ==========================================
# # 1. HELPER FUNCTIONS & METRICS
# # ==========================================
#
# def calculate_metrics(y_true, y_pred, model_name):
#     """Helper to calculate regression metrics."""
#     # Ensure inputs are 1D arrays
#     y_true = np.array(y_true).flatten()
#     y_pred = np.array(y_pred).flatten()
#
#     # Handle edge case where predictions are empty or NaN
#     if len(y_true) == 0 or len(y_pred) == 0:
#         return {
#             "name": model_name,
#             "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0
#         }
#
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
# # ==========================================
# # 2. MODEL SPECIFIC LOGIC
# # ==========================================
#
# def run_dl_model(df, symbol, model_type="LSTM"):
#     """
#     Loads saved LSTM/RNN models (which predict Balance Sheet components),
#     derives the Current Ratio from predictions, and compares to actuals.
#     """
#     # Features used in balancesheet.py for Current Ratio prediction
#     features = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
#                 'otherDebt', 'payable', 'shortDebt', 'longDebt']
#
#     # Ensure data is sorted
#     df = df.sort_values('year')
#     df = df.dropna(subset=features).reset_index(drop=True)
#
#     if len(df) < 5:
#         return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     # Data Prep
#     dataset = df[features].values.astype('float32')
#     scaler = MinMaxScaler(feature_range=(0, 1))
#
#     # Fit scaler on available data to match input distribution expected by model
#     scaler.fit(dataset)
#     scaled_data = scaler.transform(dataset)
#
#     # Sequence Creation
#     look_back = 3
#     X, y = create_sequences(scaled_data, look_back)
#
#     if len(X) == 0:
#         return {"name": model_type, "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     # Split into Train/Test (Last 20% as test)
#     split_idx = int(len(X) * 0.8)
#     X_test = X[split_idx:]
#     y_test = y[split_idx:]
#
#     # Load Model
#     if model_type == "LSTM":
#         model_path = f"trained_models/current_ratio_{symbol}.keras"
#     else:
#         # Assuming RNN model naming convention follows
#         model_path = f"trained_models/rnn_current_ratio_{symbol}.keras"
#
#     if not os.path.exists(model_path):
#         return {"name": f"{model_type} (Not Found)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     try:
#         model = load_model(model_path)
#     except Exception as e:
#         return {"name": f"{model_type} (Error)", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     # Predict
#     # X_test shape: (samples, look_back, features)
#     test_predict_scaled = model.predict(X_test, verbose=0)
#
#     # Inverse transform to get component values
#     # y_test and predictions are both (samples, 8)
#     pred_inv = scaler.inverse_transform(test_predict_scaled)
#     y_test_inv = scaler.inverse_transform(y_test)
#
#     # Calculate Current Ratio from components
#     # Indices based on features list:
#     # Assets: cash(0), shortReceivable(1), inventory(2), shortAsset(3)
#     # Liabs: otherDebt(4), payable(5), shortDebt(6), longDebt(7)
#
#     # Predicted Ratio
#     pred_current_assets = np.sum(pred_inv[:, 0:4], axis=1)
#     pred_current_liabs = np.sum(pred_inv[:, 4:8], axis=1)  # Sum of otherDebt, payable, shortDebt, longDebt
#
#     # Avoid division by zero
#     pred_ratio = np.divide(pred_current_assets, pred_current_liabs,
#                            out=np.zeros_like(pred_current_assets),
#                            where=pred_current_liabs != 0)
#
#     # Actual Ratio (Derived from inverse transformed actuals to maintain consistency)
#     true_current_assets = np.sum(y_test_inv[:, 0:4], axis=1)
#     true_current_liabs = np.sum(y_test_inv[:, 4:8], axis=1)
#
#     true_ratio = np.divide(true_current_assets, true_current_liabs,
#                            out=np.zeros_like(true_current_assets),
#                            where=true_current_liabs != 0)
#
#     return calculate_metrics(true_ratio, pred_ratio, f"{model_type} (Current Ratio)")
#
#
# def run_random_forest(df, target_col="currentRatio"):
#     # Features used for regression
#     # We use lags of the components + previous ratio
#     component_cols = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
#                       'otherDebt', 'payable', 'shortDebt', 'longDebt']
#
#     df_rf = df.copy()
#
#     # Calculate target if not present
#     if target_col not in df_rf.columns:
#         current_assets = df_rf['cash'] + df_rf['shortReceivable'] + df_rf['inventory'] + df_rf['shortAsset']
#         current_liabs = df_rf['payable'] + df_rf['shortDebt'] + df_rf['otherDebt'] + df_rf['longDebt']
#         df_rf[target_col] = current_assets / current_liabs
#         df_rf[target_col] = df_rf[target_col].replace([np.inf, -np.inf], 0).fillna(0)
#
#     # Create lag features
#     feature_cols = []
#     for col in component_cols + [target_col]:
#         df_rf[f'{col}_prev'] = df_rf[col].shift(1)
#         feature_cols.append(f'{col}_prev')
#
#     df_rf = df_rf.dropna()
#
#     X = df_rf[feature_cols].values
#     y = df_rf[target_col].values
#
#     if len(X) < 5:
#         return {"name": "Random Forest", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     # Time series split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#     rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
#     rf.fit(X_train, y_train)
#     y_pred = rf.predict(X_test)
#
#     return calculate_metrics(y_test, y_pred, "Random Forest")
#
#
# def run_linear_regression(df, target_col="currentRatio"):
#     # Same feature prep as RF
#     component_cols = ['cash', 'shortReceivable', 'inventory', 'shortAsset',
#                       'otherDebt', 'payable', 'shortDebt', 'longDebt']
#
#     df_lr = df.copy()
#
#     if target_col not in df_lr.columns:
#         current_assets = df_lr['cash'] + df_lr['shortReceivable'] + df_lr['inventory'] + df_lr['shortAsset']
#         current_liabs = df_lr['payable'] + df_lr['shortDebt'] + df_lr['otherDebt'] + df_lr['longDebt']
#         df_lr[target_col] = current_assets / current_liabs
#         df_lr[target_col] = df_lr[target_col].replace([np.inf, -np.inf], 0).fillna(0)
#
#     feature_cols = []
#     for col in component_cols + [target_col]:
#         df_lr[f'{col}_prev'] = df_lr[col].shift(1)
#         feature_cols.append(f'{col}_prev')
#
#     df_lr = df_lr.dropna()
#
#     X = df_lr[feature_cols].values
#     y = df_lr[target_col].values
#
#     if len(X) < 5:
#         return {"name": "Linear Regression", "MSE": 0, "RMSE": 0, "MAE": 0, "R2": 0}
#
#     # Temporal split
#     split_idx = int(len(X) * 0.8)
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]
#
#     lr = LinearRegression()
#     lr.fit(X_train, y_train)
#     y_pred = lr.predict(X_test)
#
#     return calculate_metrics(y_test, y_pred, "Multiple Linear Regression")
#
#
# # ==========================================
# # 3. MAIN ENDPOINT
# # ==========================================
#
# @router.get("/compare", response_class=HTMLResponse)
# async def compare_models(request: Request, session: AsyncSession = Depends(session_manager.session)):
#     """
#     Runs a benchmark comparison across LSTM, RNN, RF, and MLR
#     using REAL Balance Sheet data to predict Current Ratio.
#     """
#     symbol = "FPT"  # Default for benchmark
#
#     # 1. Fetch Real Data
#     try:
#         # Fetch data for enough years to run a test
#         df_balance = await get_balance_sheet(session, symbol, year=2023, yearly=True)
#         df_balance = df_balance.fillna(0)
#     except Exception as e:
#         return HTMLResponse(f"Error fetching data: {str(e)}")
#
#     if df_balance.empty:
#         return HTMLResponse("No data available for benchmarking.")
#
#     # 2. Run Benchmarks
#     # Target: Current Ratio
#
#     # LSTM (Loads trained_models/current_ratio_FPT.keras)
#     lstm_metrics = run_dl_model(df_balance, symbol, model_type="LSTM")
#
#     # RNN (Loads trained_models/rnn_current_ratio_FPT.keras)
#     rnn_metrics = run_dl_model(df_balance, symbol, model_type="RNN")
#
#     # ML Baselines (Predicts Current Ratio directly)
#     rf_metrics = run_random_forest(df_balance, target_col="currentRatio")
#     lr_metrics = run_linear_regression(df_balance, target_col="currentRatio")
#
#     models_data = [lstm_metrics, rnn_metrics, rf_metrics, lr_metrics]
#
#     # 3. Determine Best Model
#     valid_models = [m for m in models_data if isinstance(m['R2'], (int, float))]
#
#     best_model_name = "None"
#     if valid_models:
#         best_model = max(valid_models, key=lambda x: x['R2'])
#         best_model_name = best_model['name']
#
#     # 4. Render
#     return templates.TemplateResponse(
#         "benchmark/benchmark_testing.html",
#         {
#             "request": request,
#             "title": f"Model Performance Comparison ({symbol} - Current Ratio)",
#             "models": models_data,
#             "best_model_name": best_model_name
#         }
#     )


import math
import numpy as np
import pandas as pd
import os
from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ML/DL Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sqlalchemy.ext.asyncio import AsyncSession

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