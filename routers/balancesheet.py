import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, LeakyReLU, BatchNormalization, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

from db import session_manager
from models.balancesheet import BalanceSheet
from models.stock import Stock


router = APIRouter(
    prefix="/balance-sheet",
    tags=["balance-sheet-dashboard"]
)
templates = Jinja2Templates(directory="templates")


def create_sequences(dataset, look_back_years):
    x, y = [], []
    for i in range(len(dataset) - look_back_years):
        x.append(dataset[i:(i + look_back_years)])
        y.append(dataset[i + look_back_years])
    return np.array(x), np.array(y)


async def get_balance_sheet(session: AsyncSession, symbol: str, year: int, yearly: bool = True):
    # Get balance sheet data
    stmt_bs = (select(BalanceSheet)
               .where(BalanceSheet.yearly == True))
    queryset_bs = await session.execute(stmt_bs)
    balance_sheet_data = queryset_bs.fetchall()

    if not balance_sheet:
        raise HTTPException(status_code=404)

    # Extract data
    bs_data = []
    for row in balance_sheet_data:
        item = row[0].__dict__
        bs_data.extend(item["balance_sheet"])

    df_balance_sheet = pd.DataFrame(bs_data)
    return df_balance_sheet


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


def get_lstm_model(model_name, n_inputs, n_features, x_train, y_train, val_x, val_y):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        # Build and train model
        model = Sequential([
            Input(shape=(n_inputs, len(n_features))),
            LSTM(120, activation='relu', return_sequences=True),
            LeakyReLU(),
            GRU(50, activation='relu', return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),  # Batch Normalization layer
            LSTM(120, activation='relu'),
            LeakyReLU(),
            Dropout(0.3),
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


@router.get("/", response_class=HTMLResponse)
async def balance_sheet(
        request: Request, session: AsyncSession = Depends(session_manager.session),
        symbol: str = Query('FPT', description="Stock symbol"),
        prediction_year: int = Query(2023, description="Year to predict"),
        yearly: bool = Query(True, description="Use yearly data"),
        feature_cols_query: list[str] = Query(["asset"], description="Feature columns to query"),
):
    """
    Generate heatmap chart showing correlation matrix between balance sheet features
    """
    symbol = symbol.upper()

    # Fetch data from database
    try:
        df_balance_sheet = await get_balance_sheet(session, symbol, prediction_year, yearly)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

    feature_cols = [
        'shortAsset',
        'cash',
        'shortInvest',
        'shortReceivable',
        'inventory',
        'longAsset',
        'fixedAsset',
        'asset',
        'debt',
        'shortDebt',
        'longDebt',
        'equity',
        'capital',
        'centralBankDeposit',
        'otherBankDeposit',
        'otherBankLoan',
        'stockInvest',
        'customerLoan',
        'badLoan',
        'provision',
        'netCustomerLoan',
        'otherAsset',
        'otherBankCredit',
        'oweOtherBank',
        'oweCentralBank',
        'valuablePaper',
        'payableInterest',
        'receivableInterest',
        'deposit',
        'otherDebt',
        'fund',
        'unDistributedIncome',
        'minorShareHolderProfit',
        'payable'
    ]

    df_balance_sheet = df_balance_sheet.sort_values(['ticker', 'year']).reset_index(drop=True)
    df_balance_sheet = df_balance_sheet.fillna(value=0)

    # Process each company separately and calculate correlations
    symbols = df_balance_sheet['ticker'].unique()

    # Calculate correlation matrix for all balance sheet features
    # Filter out non-numeric columns and ensure feature columns exist
    numeric_cols = [col for col in feature_cols if col in df_balance_sheet.columns]

    if len(numeric_cols) < 2:
        # Fallback: create empty heatmap if insufficient data
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=[],
            x=[],
            y=[],
            colorscale='RdBu',
            showscale=True
        ))
        fig.update_layout(
            title='Balance Sheet Correlation Heatmap - Insufficient Data',
            height=600,
            width=1600,
        )
        plot_html = fig.to_html(full_html=False)
        return templates.TemplateResponse("balancesheet.html", {"request": request, "plot_html": plot_html})

    # Calculate correlation matrix
    corr_matrix = df_balance_sheet[numeric_cols].corr()

    # Create heatmap of correlation matrix
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title="Correlation",
            titleside="right"
        )
    ))

    # Calculate statistics for display
    # Get upper triangle values (excluding diagonal)
    corr_values = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_value = corr_matrix.iloc[i, j]
            if not pd.isna(corr_value):
                corr_values.append(corr_value)

    mean_corr = np.mean(corr_values) if corr_values else 0
    median_corr = np.median(corr_values) if corr_values else 0
    std_corr = np.std(corr_values) if corr_values else 0

    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Balance Sheet Features Correlation Heatmap<br><sub>Mean: {mean_corr:.3f}, Median: {median_corr:.3f}, Std: {std_corr:.3f}</sub><b>',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Features',
        yaxis_title='Features',
        height=600,
        width=1600,
        template='plotly_white',
    )

    correlation_metrics_html = fig.to_html(full_html=False)
    look_back = 3
    current_ratio_prediction = ['cash', 'shortReceivable', 'inventory', 'shortAsset', 'otherDebt',
                                'payable', 'shortDebt', 'longDebt']

    # Calculate current ratio components
    df_balance_sheet["currentLiabilities"] = (df_balance_sheet["payable"] +
                                              df_balance_sheet["shortDebt"] +
                                              df_balance_sheet["otherDebt"] +
                                              df_balance_sheet["longDebt"])
    df_balance_sheet["currentAsset"] = (df_balance_sheet["cash"] +
                                        df_balance_sheet["shortReceivable"] +
                                        df_balance_sheet["inventory"] +
                                        df_balance_sheet["shortAsset"])
    df_balance_sheet["currentRatio"] = df_balance_sheet["currentAsset"] / df_balance_sheet["currentLiabilities"]

    # Filter for selected symbol
    df_symbol = df_balance_sheet[df_balance_sheet['ticker'] == symbol].copy()

    # Check data sufficiency
    if len(df_symbol) < look_back + 2:
        raise HTTPException(status_code=400,
                            detail=f"Insufficient data for {symbol}. Need at least {look_back + 2} years.")

    # Prepare data
    df_symbol = df_symbol.sort_values('year')
    df_symbol = df_symbol.dropna(subset=current_ratio_prediction).reset_index(drop=True)

    if len(df_symbol) < look_back + 2:
        raise HTTPException(status_code=400,
                            detail=f"Insufficient valid data after removing NaN values.")

    data = df_symbol[current_ratio_prediction].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Get data before prediction year
    df_before_prediction = df_symbol[df_symbol['year'] < prediction_year].copy()
    data_before_prediction = df_before_prediction[current_ratio_prediction].values
    scaled_data_before_prediction = scaler.transform(data_before_prediction)

    # Create sequences
    x, y = create_sequences(scaled_data_before_prediction, look_back)
    x_train, y_train = [], []
    val_x, val_y = [], []
    if len(x) == 0:
        raise HTTPException(status_code=400, detail=f"Insufficient training data")

    if len(x) > 0:
        x_train.append(x)
        y_train.append(y)

    # Create validation sequences if validation data exists
    if len(df_symbol) > 0:
        # Use last look_back points from training + validation data
        combined_data = np.vstack([data[-look_back:], df_symbol[current_ratio_prediction].values])
        scaled_combined = scaler.transform(combined_data)

        x_val, y_val = create_sequences(scaled_combined, look_back)

        if len(x_val) > 0:
            val_x.append(x_val)
            val_y.append(y_val)

    # Check if model exists, otherwise train new one
    current_ratio_model_path = f"models/current_ratio_{symbol}.keras"
    current_ratio_model = get_lstm_model(
        current_ratio_model_path,
        len(df_balance_sheet),
        current_ratio_prediction,
        x_train=x_train,
        y_train=y_train,
        val_x=val_x,
        val_y=val_y,
    )

    # Make prediction
    last_sequence = scaled_data_before_prediction[-look_back:].copy()
    future_predictions = []

    num_years_to_predict = prediction_year - df_before_prediction['year'].max()

    for _ in range(num_years_to_predict):
        pred_input = last_sequence.reshape(1, look_back, len(current_ratio_prediction))
        pred = current_ratio_model.predict(pred_input, verbose=0)
        future_predictions.append(pred[0])
        last_sequence = np.vstack([last_sequence[1:], pred[0]])

    # Inverse transform predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions))

    # Calculate predicted current ratio
    predicted_values = future_predictions[-1]
    predicted_currentAsset = (predicted_values[0] + predicted_values[1] +
                              predicted_values[2] + predicted_values[3] + predicted_values[4])
    predicted_currentLiabilities = (predicted_values[5] + predicted_values[6] + predicted_values[7])
    predicted_currentRatio = predicted_currentAsset / predicted_currentLiabilities if predicted_currentLiabilities != 0 else 0

    # Build annotation
    annotation_text = f"<b>{prediction_year} Current Ratio Prediction for {symbol}</b><br>"
    annotation_text += f"<b>Predicted Current Ratio</b>: {predicted_currentRatio:.2f}<br><br>"
    annotation_text += f"<b>Predicted Components:</b><br>"
    annotation_text += f"• Current Asset: {predicted_currentAsset:,.2f}B VND<br>"
    annotation_text += f"• Current Liabilities: {predicted_currentLiabilities:,.2f}B VND<br><br>"

    # Check if actual data exists
    actual_prediction_mask = df_symbol['year'] == prediction_year
    actual_currentRatio = None

    if actual_prediction_mask.any():
        actual_currentRatio = df_symbol[actual_prediction_mask]['currentRatio'].values[0]
        difference = predicted_currentRatio - actual_currentRatio
        percentage_diff = (difference / actual_currentRatio) * 100 if actual_currentRatio != 0 else 0

        annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        annotation_text += f"• Actual {prediction_year}: {actual_currentRatio:.2f}<br>"
        annotation_text += f"• Difference: {difference:.2f} ({percentage_diff:+.2f}%)<br><br>"

    # Add liquidity assessment
    annotation_text += f"<b>Predicted Liquidity Assessment:</b><br>"
    if predicted_currentRatio >= 2.0:
        annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
        annotation_text += f"• Company has strong ability to cover short-term obligations<br>"
        annotation_text += f"• Current assets are {predicted_currentRatio:.2f}x current liabilities<br>"
    elif predicted_currentRatio >= 1.5:
        annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
        annotation_text += f"• Company can comfortably meet short-term obligations<br>"
        annotation_text += f"• Healthy liquidity position maintained<br>"
    elif predicted_currentRatio >= 1.0:
        annotation_text += f"• Expected Status: <b style='color:orange'>Adequate</b><br>"
        annotation_text += f"• Company can meet obligations but with limited buffer<br>"
        annotation_text += f"• Monitor closely for liquidity issues<br>"
    else:
        annotation_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
        annotation_text += f"• Current assets insufficient to cover current liabilities<br>"
        annotation_text += f"• Potential liquidity concerns - ratio below 1.0<br>"

    # Create visualization
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['currentRatio'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current Ratio: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    fig.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_currentRatio],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=12, symbol='square'),
            hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        )
    )

    # Actual value if available
    if actual_prediction_mask.any():
        fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[actual_currentRatio],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Warning Level (1.0)", annotation_position="top")
    fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="Healthy Level (1.5)", annotation_position="top")

    # Update layout
    fig.update_layout(
        title=f'LSTM Current Ratio Trends - Ability to Cover Short-Term Debt and Bills - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Ratio',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    current_ratio_metrics_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # RNN model
    # Check if model exists, otherwise train new one
    rnn_current_ratio_model_path = f"models/rnn_current_ratio_{symbol}.keras"
    rnn_current_ratio_model = get_rnn_model(
        rnn_current_ratio_model_path,
        len(df_balance_sheet),
        current_ratio_prediction,
        x_train=x_train,
        y_train=y_train,
        val_x=val_x,
        val_y=val_y,
    )

    # Make prediction
    last_sequence = scaled_data_before_prediction[-look_back:].copy()
    rnn_predictions = []

    num_years_to_predict = prediction_year - df_before_prediction['year'].max()

    for _ in range(num_years_to_predict):
        pred_input = last_sequence.reshape(1, look_back, len(current_ratio_prediction))
        pred = rnn_current_ratio_model.predict(pred_input, verbose=0)
        rnn_predictions.append(pred[0])
        last_sequence = np.vstack([last_sequence[1:], pred[0]])

    # Inverse transform predictions
    rnn_predictions = scaler.inverse_transform(np.array(rnn_predictions))

    # Calculate predicted current ratio
    rnn_predicted_values = rnn_predictions[-1]
    rnn_predicted_current_asset = (rnn_predicted_values[0] + rnn_predicted_values[1] +
                              rnn_predicted_values[2] + rnn_predicted_values[3] + rnn_predicted_values[4])
    rnn_predicted_current_liabilities = (rnn_predicted_values[5] + rnn_predicted_values[6] + rnn_predicted_values[7])
    rnn_predicted_current_ratio = rnn_predicted_current_asset / rnn_predicted_current_liabilities if rnn_predicted_current_liabilities != 0 else 0

    # Build annotation
    rnn_annotation_text = f"<b>{prediction_year} Current Ratio Prediction for {symbol}</b><br>"
    rnn_annotation_text += f"<b>Predicted Current Ratio</b>: {rnn_predicted_current_ratio:.2f}<br><br>"
    rnn_annotation_text += f"<b>Predicted Components:</b><br>"
    rnn_annotation_text += f"• Current Asset: {rnn_predicted_current_asset:,.2f}B VND<br>"
    rnn_annotation_text += f"• Current Liabilities: {rnn_predicted_current_liabilities:,.2f}B VND<br><br>"

    # Check if actual data exists
    actual_prediction_mask = df_symbol['year'] == prediction_year
    actual_currentRatio = None

    if actual_prediction_mask.any():
        actual_currentRatio = df_symbol[actual_prediction_mask]['currentRatio'].values[0]
        difference = rnn_predicted_current_ratio - actual_currentRatio
        percentage_diff = (difference / actual_currentRatio) * 100 if actual_currentRatio != 0 else 0

        rnn_annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        rnn_annotation_text += f"• Actual {prediction_year}: {actual_currentRatio:.2f}<br>"
        rnn_annotation_text += f"• Difference: {difference:.2f} ({percentage_diff:+.2f}%)<br><br>"

    # Add liquidity assessment
    rnn_annotation_text += f"<b>Predicted Liquidity Assessment:</b><br>"
    if rnn_predicted_current_ratio >= 2.0:
        rnn_annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
        rnn_annotation_text += f"• Company has strong ability to cover short-term obligations<br>"
        rnn_annotation_text += f"• Current assets are {rnn_predicted_current_ratio:.2f}x current liabilities<br>"
    elif rnn_predicted_current_ratio >= 1.5:
        rnn_annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
        rnn_annotation_text += f"• Company can comfortably meet short-term obligations<br>"
        rnn_annotation_text += f"• Healthy liquidity position maintained<br>"
    elif rnn_predicted_current_ratio >= 1.0:
        rnn_annotation_text += f"• Expected Status: <b style='color:orange'>Adequate</b><br>"
        rnn_annotation_text += f"• Company can meet obligations but with limited buffer<br>"
        rnn_annotation_text += f"• Monitor closely for liquidity issues<br>"
    else:
        rnn_annotation_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
        rnn_annotation_text += f"• Current assets insufficient to cover current liabilities<br>"
        rnn_annotation_text += f"• Potential liquidity concerns - ratio below 1.0<br>"

    # Create visualization
    rnn_fig = go.Figure()

    # Historical data
    rnn_fig.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['currentRatio'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current Ratio: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    rnn_fig.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[rnn_predicted_current_ratio],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=12, symbol='square'),
            hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        )
    )

    # Actual value if available
    if actual_prediction_mask.any():
        rnn_fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[actual_currentRatio],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Add reference lines
    rnn_fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Warning Level (1.0)", annotation_position="top")
    rnn_fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="Healthy Level (1.5)", annotation_position="top")

    # Update layout
    rnn_fig.update_layout(
        title=f'RNN Model Current Ratio Trends - Ability to Cover Short-Term Debt and Bills - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Ratio',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    rnn_current_ratio_metrics_html = rnn_fig.to_html(full_html=False, include_plotlyjs='cdn')

    symbols = await get_symbols(session)
    context = {
        "request": request,
        "correlation_metrics_html": correlation_metrics_html,
        "current_ratio_metrics_html": current_ratio_metrics_html,
        "current_ratio_metrics_text": annotation_text,
        "rnn_current_ratio_metrics_html": rnn_current_ratio_metrics_html,
        "rnn_annotation_text": rnn_annotation_text,
        "features_2_predict": feature_cols,
        "symbols": symbols,
        "symbol": symbol,
    }

    return templates.TemplateResponse("balancesheet.html", context=context)
