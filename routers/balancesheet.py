import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sqlalchemy.ext.asyncio import AsyncSession

from core.helpers import (
    create_sequences,
    get_lstm_model,
    get_symbols,
    get_balance_sheet,
    get_rnn_model, train_and_predict_ratio_random_forest,
)
from db import session_manager

router = APIRouter(
    prefix="/balance-sheet",
    tags=["balance-sheet-dashboard"]
)
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def balance_sheet(
        request: Request, session: AsyncSession = Depends(session_manager.session),
        symbol: str = Query('FPT', description="Stock symbol"),
        model_type: str = Query('LSTM', description="Model to use for prediction"),
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
    # Calculate summary statistics
    summary = {}

    # Latest values
    if len(df_balance_sheet) > 0:
        summary['latest_asset'] = float(df_balance_sheet['asset'].iloc[-1]) if 'asset' in df_balance_sheet.columns else 0
        summary['latest_debt'] = float(df_balance_sheet['debt'].iloc[-1]) if 'debt' in df_balance_sheet.columns else 0
        summary['latest_equity'] = float(df_balance_sheet['equity'].iloc[-1]) if 'equity' in df_balance_sheet.columns else 0

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
    
    # Calculate quick ratio: (shortAsset - inventory) / shortDebt
    df_balance_sheet["quickRatio"] = (df_balance_sheet["shortAsset"] - df_balance_sheet["inventory"]) / df_balance_sheet["shortDebt"]
    # Replace inf and NaN with 0 for quick ratio
    df_balance_sheet["quickRatio"] = df_balance_sheet["quickRatio"].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate debt ratio: debt / equity
    df_balance_sheet["debtRatio"] = df_balance_sheet["debt"] / df_balance_sheet["equity"]
    # Replace inf and NaN with 0 for debt ratio
    df_balance_sheet["debtRatio"] = df_balance_sheet["debtRatio"].replace([np.inf, -np.inf], 0).fillna(0)

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
    current_ratio_model_path = f"trained_models/current_ratio_{symbol}.keras"
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
    rnn_current_ratio_model_path = f"trained_models/rnn_current_ratio_{symbol}.keras"
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
                  annotation_text="Danger Zone (1.0)", annotation_position="top")
    rnn_fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                  annotation_text="Very Safe (1.5)", annotation_position="top")

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

    # ========== QUICK RATIO PREDICTION ==========
    # Quick ratio features: shortAsset, inventory, shortDebt and related features
    quick_ratio_prediction = ['shortAsset', 'inventory', 'shortDebt', 'cash', 'shortReceivable', 'payable', 'otherDebt']
    
    # Prepare data for quick ratio
    df_symbol_quick = df_balance_sheet[df_balance_sheet['ticker'] == symbol].copy()
    df_symbol_quick = df_symbol_quick.sort_values('year')
    df_symbol_quick = df_symbol_quick.dropna(subset=quick_ratio_prediction).reset_index(drop=True)
    
    if len(df_symbol_quick) < look_back + 2:
        # If insufficient data, create empty quick ratio visualizations
        quick_ratio_metrics_html = go.Figure().to_html(full_html=False, include_plotlyjs='cdn')
        quick_ratio_annotation_text = f"<b>Insufficient data for Quick Ratio prediction</b>"
        rnn_quick_ratio_metrics_html = go.Figure().to_html(full_html=False, include_plotlyjs='cdn')
        rnn_quick_ratio_annotation_text = f"<b>Insufficient data for Quick Ratio prediction</b>"
    else:
        quick_data = df_symbol_quick[quick_ratio_prediction].values
        
        # Normalize data
        quick_scaler = MinMaxScaler(feature_range=(0, 1))
        quick_scaled_data = quick_scaler.fit_transform(quick_data)
        
        # Get data before prediction year
        df_before_prediction_quick = df_symbol_quick[df_symbol_quick['year'] < prediction_year].copy()
        data_before_prediction_quick = df_before_prediction_quick[quick_ratio_prediction].values
        scaled_data_before_prediction_quick = quick_scaler.transform(data_before_prediction_quick)
        
        # Create sequences
        quick_x, quick_y = create_sequences(scaled_data_before_prediction_quick, look_back)
        quick_x_train, quick_y_train = [], []
        quick_val_x, quick_val_y = [], []
        
        if len(quick_x) > 0:
            quick_x_train.append(quick_x)
            quick_y_train.append(quick_y)
        
        # Create validation sequences
        if len(df_symbol_quick) > 0:
            combined_data_quick = np.vstack([quick_data[-look_back:], df_symbol_quick[quick_ratio_prediction].values])
            scaled_combined_quick = quick_scaler.transform(combined_data_quick)
            quick_x_val, quick_y_val = create_sequences(scaled_combined_quick, look_back)
            
            if len(quick_x_val) > 0:
                quick_val_x.append(quick_x_val)
                quick_val_y.append(quick_y_val)
        
        # LSTM model for quick ratio
        quick_ratio_model_path = f"trained_models/quick_ratio_{symbol}.keras"
        quick_ratio_model = get_lstm_model(
            quick_ratio_model_path,
            len(df_balance_sheet),
            quick_ratio_prediction,
            x_train=quick_x_train,
            y_train=quick_y_train,
            val_x=quick_val_x,
            val_y=quick_val_y,
        )
        
        # Make prediction
        last_sequence_quick = scaled_data_before_prediction_quick[-look_back:].copy()
        future_predictions_quick = []
        
        num_years_to_predict_quick = prediction_year - df_before_prediction_quick['year'].max()
        
        for _ in range(num_years_to_predict_quick):
            pred_input_quick = last_sequence_quick.reshape(1, look_back, len(quick_ratio_prediction))
            pred_quick = quick_ratio_model.predict(pred_input_quick, verbose=0)
            future_predictions_quick.append(pred_quick[0])
            last_sequence_quick = np.vstack([last_sequence_quick[1:], pred_quick[0]])
        
        # Inverse transform predictions
        future_predictions_quick = quick_scaler.inverse_transform(np.array(future_predictions_quick))
        
        # Calculate predicted quick ratio
        predicted_values_quick = future_predictions_quick[-1]
        predicted_quick_asset = predicted_values_quick[0] - predicted_values_quick[1]  # shortAsset - inventory
        predicted_short_debt = predicted_values_quick[2]
        predicted_quick_ratio = predicted_quick_asset / predicted_short_debt if predicted_short_debt != 0 else 0
        
        # Build annotation
        quick_ratio_annotation_text = f"<b>{prediction_year} Quick Ratio Prediction for {symbol}</b><br>"
        quick_ratio_annotation_text += f"<b>Predicted Quick Ratio</b>: {predicted_quick_ratio:.2f}<br><br>"
        quick_ratio_annotation_text += f"<b>Predicted Components:</b><br>"
        quick_ratio_annotation_text += f"• Quick Assets (Short Asset - Inventory): {predicted_quick_asset:,.2f}B VND<br>"
        quick_ratio_annotation_text += f"• Short Debt: {predicted_short_debt:,.2f}B VND<br><br>"
        
        # Check if actual data exists
        actual_quick_prediction_mask = df_symbol_quick['year'] == prediction_year
        actual_quick_ratio = None
        
        if actual_quick_prediction_mask.any():
            actual_quick_ratio = df_symbol_quick[actual_quick_prediction_mask]['quickRatio'].values[0]
            difference_quick = predicted_quick_ratio - actual_quick_ratio
            percentage_diff_quick = (difference_quick / actual_quick_ratio) * 100 if actual_quick_ratio != 0 else 0
            
            quick_ratio_annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
            quick_ratio_annotation_text += f"• Actual {prediction_year}: {actual_quick_ratio:.2f}<br>"
            quick_ratio_annotation_text += f"• Difference: {difference_quick:.2f} ({percentage_diff_quick:+.2f}%)<br><br>"
        
        # Add liquidity assessment
        quick_ratio_annotation_text += f"<b>Predicted Liquidity Assessment:</b><br>"
        if predicted_quick_ratio >= 1.5:
            quick_ratio_annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
            quick_ratio_annotation_text += f"• Company has strong ability to cover short-term debt without inventory<br>"
            quick_ratio_annotation_text += f"• Quick assets are {predicted_quick_ratio:.2f}x short-term debt<br>"
        elif predicted_quick_ratio >= 1.0:
            quick_ratio_annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
            quick_ratio_annotation_text += f"• Company can comfortably meet short-term debt obligations<br>"
            quick_ratio_annotation_text += f"• Healthy quick liquidity position maintained<br>"
        elif predicted_quick_ratio >= 0.5:
            quick_ratio_annotation_text += f"• Expected Status: <b style='color:orange'>Adequate</b><br>"
            quick_ratio_annotation_text += f"• Company can meet obligations but with limited buffer<br>"
            quick_ratio_annotation_text += f"• Monitor closely for liquidity issues<br>"
        else:
            quick_ratio_annotation_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            quick_ratio_annotation_text += f"• Quick assets insufficient to cover short-term debt<br>"
            quick_ratio_annotation_text += f"• Potential liquidity concerns - ratio below 0.5<br>"
        
        # Create visualization
        quick_fig = go.Figure()
        
        # Historical data
        quick_fig.add_trace(
            go.Scatter(
                x=df_symbol_quick['year'],
                y=df_symbol_quick['quickRatio'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='purple', width=2),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Quick Ratio: %{y:.2f}<extra></extra>'
            )
        )
        
        # Prediction
        quick_fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[predicted_quick_ratio],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=12, symbol='square'),
                hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
            )
        )
        
        # Actual value if available
        if actual_quick_prediction_mask.any():
            quick_fig.add_trace(
                go.Scatter(
                    x=[prediction_year],
                    y=[actual_quick_ratio],
                    mode='markers',
                    name=f'Actual {prediction_year}',
                    marker=dict(color='green', size=14, symbol='diamond'),
                    hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                )
            )
        
        # Add reference lines
        quick_fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                          annotation_text="Danger Zone (1.0)", annotation_position="top")
        quick_fig.add_hline(y=1.5, line_dash="dash", line_color="green",
                          annotation_text="Very Safe (1.5)", annotation_position="top")
        
        # Update layout
        quick_fig.update_layout(
            title=f'LSTM Quick Ratio Trends - Ability to Cover Short-Term Debt (Excluding Inventory) - {symbol}',
            xaxis_title='Year',
            yaxis_title='Quick Ratio',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Convert plot to HTML
        quick_ratio_metrics_html = quick_fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # RNN model for quick ratio
        rnn_quick_ratio_model_path = f"trained_models/rnn_quick_ratio_{symbol}.keras"
        rnn_quick_ratio_model = get_rnn_model(
            rnn_quick_ratio_model_path,
            len(df_balance_sheet),
            quick_ratio_prediction,
            x_train=quick_x_train,
            y_train=quick_y_train,
            val_x=quick_val_x,
            val_y=quick_val_y,
        )
        
        # Make prediction
        last_sequence_rnn_quick = scaled_data_before_prediction_quick[-look_back:].copy()
        rnn_predictions_quick = []
        
        for _ in range(num_years_to_predict_quick):
            pred_input_rnn_quick = last_sequence_rnn_quick.reshape(1, look_back, len(quick_ratio_prediction))
            pred_rnn_quick = rnn_quick_ratio_model.predict(pred_input_rnn_quick, verbose=0)
            rnn_predictions_quick.append(pred_rnn_quick[0])
            last_sequence_rnn_quick = np.vstack([last_sequence_rnn_quick[1:], pred_rnn_quick[0]])
        
        # Inverse transform predictions
        rnn_predictions_quick = quick_scaler.inverse_transform(np.array(rnn_predictions_quick))
        
        # Calculate predicted quick ratio
        rnn_predicted_values_quick = rnn_predictions_quick[-1]
        rnn_predicted_quick_asset = rnn_predicted_values_quick[0] - rnn_predicted_values_quick[1]  # shortAsset - inventory
        rnn_predicted_short_debt = rnn_predicted_values_quick[2]
        rnn_predicted_quick_ratio = rnn_predicted_quick_asset / rnn_predicted_short_debt if rnn_predicted_short_debt != 0 else 0
        
        # Build annotation
        rnn_quick_ratio_annotation_text = f"<b>{prediction_year} Quick Ratio Prediction for {symbol}</b><br>"
        rnn_quick_ratio_annotation_text += f"<b>Predicted Quick Ratio</b>: {rnn_predicted_quick_ratio:.2f}<br><br>"
        rnn_quick_ratio_annotation_text += f"<b>Predicted Components:</b><br>"
        rnn_quick_ratio_annotation_text += f"• Quick Assets (Short Asset - Inventory): {rnn_predicted_quick_asset:,.2f}B VND<br>"
        rnn_quick_ratio_annotation_text += f"• Short Debt: {rnn_predicted_short_debt:,.2f}B VND<br><br>"
        
        # Check if actual data exists
        if actual_quick_prediction_mask.any():
            actual_quick_ratio = df_symbol_quick[actual_quick_prediction_mask]['quickRatio'].values[0]
            difference_rnn_quick = rnn_predicted_quick_ratio - actual_quick_ratio
            percentage_diff_rnn_quick = (difference_rnn_quick / actual_quick_ratio) * 100 if actual_quick_ratio != 0 else 0
            
            rnn_quick_ratio_annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
            rnn_quick_ratio_annotation_text += f"• Actual {prediction_year}: {actual_quick_ratio:.2f}<br>"
            rnn_quick_ratio_annotation_text += f"• Difference: {difference_rnn_quick:.2f} ({percentage_diff_rnn_quick:+.2f}%)<br><br>"
        
        # Add liquidity assessment
        rnn_quick_ratio_annotation_text += f"<b>Predicted Liquidity Assessment:</b><br>"
        if rnn_predicted_quick_ratio >= 1.5:
            rnn_quick_ratio_annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
            rnn_quick_ratio_annotation_text += f"• Company has strong ability to cover short-term debt without inventory<br>"
            rnn_quick_ratio_annotation_text += f"• Quick assets are {rnn_predicted_quick_ratio:.2f}x short-term debt<br>"
        elif rnn_predicted_quick_ratio >= 1.0:
            rnn_quick_ratio_annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
            rnn_quick_ratio_annotation_text += f"• Company can comfortably meet short-term debt obligations<br>"
            rnn_quick_ratio_annotation_text += f"• Healthy quick liquidity position maintained<br>"
        elif rnn_predicted_quick_ratio >= 0.5:
            rnn_quick_ratio_annotation_text += f"• Expected Status: <b style='color:orange'>Adequate</b><br>"
            rnn_quick_ratio_annotation_text += f"• Company can meet obligations but with limited buffer<br>"
            rnn_quick_ratio_annotation_text += f"• Monitor closely for liquidity issues<br>"
        else:
            rnn_quick_ratio_annotation_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            rnn_quick_ratio_annotation_text += f"• Quick assets insufficient to cover short-term debt<br>"
            rnn_quick_ratio_annotation_text += f"• Potential liquidity concerns - ratio below 0.5<br>"
        
        # Create visualization
        rnn_quick_fig = go.Figure()
        
        # Historical data
        rnn_quick_fig.add_trace(
            go.Scatter(
                x=df_symbol_quick['year'],
                y=df_symbol_quick['quickRatio'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='purple', width=2),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Quick Ratio: %{y:.2f}<extra></extra>'
            )
        )
        
        # Prediction
        rnn_quick_fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[rnn_predicted_quick_ratio],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=12, symbol='square'),
                hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
            )
        )
        
        # Actual value if available
        if actual_quick_prediction_mask.any():
            rnn_quick_fig.add_trace(
                go.Scatter(
                    x=[prediction_year],
                    y=[actual_quick_ratio],
                    mode='markers',
                    name=f'Actual {prediction_year}',
                    marker=dict(color='green', size=14, symbol='diamond'),
                    hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                )
            )
        
        # Add reference lines
        rnn_quick_fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                              annotation_text="Warning Level (0.5)", annotation_position="top")
        rnn_quick_fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                              annotation_text="Healthy Level (1.0)", annotation_position="top")
        
        # Update layout
        rnn_quick_fig.update_layout(
            title=f'RNN Model Quick Ratio Trends - Ability to Cover Short-Term Debt (Excluding Inventory) - {symbol}',
            xaxis_title='Year',
            yaxis_title='Quick Ratio',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Convert plot to HTML
        rnn_quick_ratio_metrics_html = rnn_quick_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # ========== DEBT RATIO PREDICTION ==========
    # Debt ratio features: debt, equity and related features
    debt_ratio_prediction = ['debt', 'equity', 'shortDebt', 'longDebt', 'capital', 'asset', 'otherDebt']
    
    # Prepare data for debt ratio
    df_symbol_debt = df_balance_sheet[df_balance_sheet['ticker'] == symbol].copy()
    df_symbol_debt = df_symbol_debt.sort_values('year')
    df_symbol_debt = df_symbol_debt.dropna(subset=debt_ratio_prediction).reset_index(drop=True)
    
    if len(df_symbol_debt) < look_back + 2:
        # If insufficient data, create empty debt ratio visualizations
        debt_ratio_metrics_html = go.Figure().to_html(full_html=False, include_plotlyjs='cdn')
        debt_ratio_annotation_text = f"<b>Insufficient data for Debt Ratio prediction</b>"
        rnn_debt_ratio_metrics_html = go.Figure().to_html(full_html=False, include_plotlyjs='cdn')
        rnn_debt_ratio_annotation_text = f"<b>Insufficient data for Debt Ratio prediction</b>"
    else:
        debt_data = df_symbol_debt[debt_ratio_prediction].values
        
        # Normalize data
        debt_scaler = MinMaxScaler(feature_range=(0, 1))
        debt_scaled_data = debt_scaler.fit_transform(debt_data)
        
        # Get data before prediction year
        df_before_prediction_debt = df_symbol_debt[df_symbol_debt['year'] < prediction_year].copy()
        data_before_prediction_debt = df_before_prediction_debt[debt_ratio_prediction].values
        scaled_data_before_prediction_debt = debt_scaler.transform(data_before_prediction_debt)
        
        # Create sequences
        debt_x, debt_y = create_sequences(scaled_data_before_prediction_debt, look_back)
        debt_x_train, debt_y_train = [], []
        debt_val_x, debt_val_y = [], []
        
        if len(debt_x) > 0:
            debt_x_train.append(debt_x)
            debt_y_train.append(debt_y)
        
        # Create validation sequences
        if len(df_symbol_debt) > 0:
            combined_data_debt = np.vstack([debt_data[-look_back:], df_symbol_debt[debt_ratio_prediction].values])
            scaled_combined_debt = debt_scaler.transform(combined_data_debt)
            debt_x_val, debt_y_val = create_sequences(scaled_combined_debt, look_back)
            
            if len(debt_x_val) > 0:
                debt_val_x.append(debt_x_val)
                debt_val_y.append(debt_y_val)
        
        # LSTM model for debt ratio
        debt_ratio_model_path = f"trained_models/debt_ratio_{symbol}.keras"
        debt_ratio_model = get_lstm_model(
            debt_ratio_model_path,
            len(df_balance_sheet),
            debt_ratio_prediction,
            x_train=debt_x_train,
            y_train=debt_y_train,
            val_x=debt_val_x,
            val_y=debt_val_y,
        )
        
        # Make prediction
        last_sequence_debt = scaled_data_before_prediction_debt[-look_back:].copy()
        future_predictions_debt = []
        
        num_years_to_predict_debt = prediction_year - df_before_prediction_debt['year'].max()
        
        for _ in range(num_years_to_predict_debt):
            pred_input_debt = last_sequence_debt.reshape(1, look_back, len(debt_ratio_prediction))
            pred_debt = debt_ratio_model.predict(pred_input_debt, verbose=0)
            future_predictions_debt.append(pred_debt[0])
            last_sequence_debt = np.vstack([last_sequence_debt[1:], pred_debt[0]])
        
        # Inverse transform predictions
        future_predictions_debt = debt_scaler.inverse_transform(np.array(future_predictions_debt))
        
        # Calculate predicted debt ratio
        predicted_values_debt = future_predictions_debt[-1]
        predicted_debt = predicted_values_debt[0]
        predicted_equity = predicted_values_debt[1]
        predicted_debt_ratio = predicted_debt / predicted_equity if predicted_equity != 0 else 0
        
        # Build annotation
        debt_ratio_annotation_text = f"<b>{prediction_year} Debt Ratio Prediction for {symbol}</b><br>"
        debt_ratio_annotation_text += f"<b>Predicted Debt Ratio</b>: {predicted_debt_ratio:.2f}<br><br>"
        debt_ratio_annotation_text += f"<b>Predicted Components:</b><br>"
        debt_ratio_annotation_text += f"• Total Debt: {predicted_debt:,.2f}B VND<br>"
        debt_ratio_annotation_text += f"• Total Equity: {predicted_equity:,.2f}B VND<br><br>"
        
        # Check if actual data exists
        actual_debt_prediction_mask = df_symbol_debt['year'] == prediction_year
        actual_debt_ratio = None
        
        if actual_debt_prediction_mask.any():
            actual_debt_ratio = df_symbol_debt[actual_debt_prediction_mask]['debtRatio'].values[0]
            difference_debt = predicted_debt_ratio - actual_debt_ratio
            percentage_diff_debt = (difference_debt / actual_debt_ratio) * 100 if actual_debt_ratio != 0 else 0
            
            debt_ratio_annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
            debt_ratio_annotation_text += f"• Actual {prediction_year}: {actual_debt_ratio:.2f}<br>"
            debt_ratio_annotation_text += f"• Difference: {difference_debt:.2f} ({percentage_diff_debt:+.2f}%)<br><br>"
        
        # Add leverage assessment
        debt_ratio_annotation_text += f"<b>Predicted Leverage Assessment:</b><br>"
        if predicted_debt_ratio <= 0.3:
            debt_ratio_annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
            debt_ratio_annotation_text += f"• Company has low debt relative to equity<br>"
            debt_ratio_annotation_text += f"• Strong financial position with minimal leverage<br>"
        elif predicted_debt_ratio <= 0.5:
            debt_ratio_annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
            debt_ratio_annotation_text += f"• Company maintains moderate debt levels<br>"
            debt_ratio_annotation_text += f"• Healthy balance between debt and equity<br>"
        elif predicted_debt_ratio <= 0.7:
            debt_ratio_annotation_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
            debt_ratio_annotation_text += f"• Company has moderate to high leverage<br>"
            debt_ratio_annotation_text += f"• Monitor debt levels and repayment capacity<br>"
        else:
            debt_ratio_annotation_text += f"• Expected Status: <b style='color:red'>High Risk</b><br>"
            debt_ratio_annotation_text += f"• Company has high debt relative to equity<br>"
            debt_ratio_annotation_text += f"• Potential financial stress - debt exceeds equity significantly<br>"
        
        # Create visualization
        debt_fig = go.Figure()
        
        # Historical data
        debt_fig.add_trace(
            go.Scatter(
                x=df_symbol_debt['year'],
                y=df_symbol_debt['debtRatio'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='orange', width=2),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Debt Ratio: %{y:.2f}<extra></extra>'
            )
        )
        
        # Prediction
        debt_fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[predicted_debt_ratio],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=12, symbol='square'),
                hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
            )
        )
        
        # Actual value if available
        if actual_debt_prediction_mask.any():
            debt_fig.add_trace(
                go.Scatter(
                    x=[prediction_year],
                    y=[actual_debt_ratio],
                    mode='markers',
                    name=f'Actual {prediction_year}',
                    marker=dict(color='green', size=14, symbol='diamond'),
                    hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                )
            )
        
        # Add reference lines
        debt_fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                          annotation_text="Relative Safe (1.0)", annotation_position="bottom")
        debt_fig.add_hline(y=2.0, line_dash="dash", line_color="orange",
                          annotation_text="Risky (2.0)", annotation_position="top")
        debt_fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                          annotation_text="Extremely Dangerous (5.0)", annotation_position="top")
        
        # Update layout
        debt_fig.update_layout(
            title=f'LSTM Debt Ratio Trends - Debt to Equity Ratio - {symbol}',
            xaxis_title='Year',
            yaxis_title='Debt Ratio',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Convert plot to HTML
        debt_ratio_metrics_html = debt_fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # RNN model for debt ratio
        rnn_debt_ratio_model_path = f"trained_models/rnn_debt_ratio_{symbol}.keras"
        rnn_debt_ratio_model = get_rnn_model(
            rnn_debt_ratio_model_path,
            len(df_balance_sheet),
            debt_ratio_prediction,
            x_train=debt_x_train,
            y_train=debt_y_train,
            val_x=debt_val_x,
            val_y=debt_val_y,
        )
        
        # Make prediction
        last_sequence_rnn_debt = scaled_data_before_prediction_debt[-look_back:].copy()
        rnn_predictions_debt = []
        
        for _ in range(num_years_to_predict_debt):
            pred_input_rnn_debt = last_sequence_rnn_debt.reshape(1, look_back, len(debt_ratio_prediction))
            pred_rnn_debt = rnn_debt_ratio_model.predict(pred_input_rnn_debt, verbose=0)
            rnn_predictions_debt.append(pred_rnn_debt[0])
            last_sequence_rnn_debt = np.vstack([last_sequence_rnn_debt[1:], pred_rnn_debt[0]])
        
        # Inverse transform predictions
        rnn_predictions_debt = debt_scaler.inverse_transform(np.array(rnn_predictions_debt))
        
        # Calculate predicted debt ratio
        rnn_predicted_values_debt = rnn_predictions_debt[-1]
        rnn_predicted_debt = rnn_predicted_values_debt[0]
        rnn_predicted_equity = rnn_predicted_values_debt[1]
        rnn_predicted_debt_ratio = rnn_predicted_debt / rnn_predicted_equity if rnn_predicted_equity != 0 else 0
        
        # Build annotation
        rnn_debt_ratio_annotation_text = f"<b>{prediction_year} Debt Ratio Prediction for {symbol}</b><br>"
        rnn_debt_ratio_annotation_text += f"<b>Predicted Debt Ratio</b>: {rnn_predicted_debt_ratio:.2f}<br><br>"
        rnn_debt_ratio_annotation_text += f"<b>Predicted Components:</b><br>"
        rnn_debt_ratio_annotation_text += f"• Total Debt: {rnn_predicted_debt:,.2f}B VND<br>"
        rnn_debt_ratio_annotation_text += f"• Total Equity: {rnn_predicted_equity:,.2f}B VND<br><br>"
        
        # Check if actual data exists
        if actual_debt_prediction_mask.any():
            actual_debt_ratio = df_symbol_debt[actual_debt_prediction_mask]['debtRatio'].values[0]
            difference_rnn_debt = rnn_predicted_debt_ratio - actual_debt_ratio
            percentage_diff_rnn_debt = (difference_rnn_debt / actual_debt_ratio) * 100 if actual_debt_ratio != 0 else 0
            
            rnn_debt_ratio_annotation_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
            rnn_debt_ratio_annotation_text += f"• Actual {prediction_year}: {actual_debt_ratio:.2f}<br>"
            rnn_debt_ratio_annotation_text += f"• Difference: {difference_rnn_debt:.2f} ({percentage_diff_rnn_debt:+.2f}%)<br><br>"
        
        # Add leverage assessment
        rnn_debt_ratio_annotation_text += f"<b>Predicted Leverage Assessment:</b><br>"
        if rnn_predicted_debt_ratio <= 0.3:
            rnn_debt_ratio_annotation_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
            rnn_debt_ratio_annotation_text += f"• Company has low debt relative to equity<br>"
            rnn_debt_ratio_annotation_text += f"• Strong financial position with minimal leverage<br>"
        elif rnn_predicted_debt_ratio <= 0.5:
            rnn_debt_ratio_annotation_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
            rnn_debt_ratio_annotation_text += f"• Company maintains moderate debt levels<br>"
            rnn_debt_ratio_annotation_text += f"• Healthy balance between debt and equity<br>"
        elif rnn_predicted_debt_ratio <= 0.7:
            rnn_debt_ratio_annotation_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
            rnn_debt_ratio_annotation_text += f"• Company has moderate to high leverage<br>"
            rnn_debt_ratio_annotation_text += f"• Monitor debt levels and repayment capacity<br>"
        else:
            rnn_debt_ratio_annotation_text += f"• Expected Status: <b style='color:red'>High Risk</b><br>"
            rnn_debt_ratio_annotation_text += f"• Company has high debt relative to equity<br>"
            rnn_debt_ratio_annotation_text += f"• Potential financial stress - debt exceeds equity significantly<br>"
        
        # Create visualization
        rnn_debt_fig = go.Figure()
        
        # Historical data
        rnn_debt_fig.add_trace(
            go.Scatter(
                x=df_symbol_debt['year'],
                y=df_symbol_debt['debtRatio'],
                mode='lines+markers',
                name='Historical',
                line=dict(color='orange', width=2),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Debt Ratio: %{y:.2f}<extra></extra>'
            )
        )
        
        # Prediction
        rnn_debt_fig.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[rnn_predicted_debt_ratio],
                mode='markers',
                name='Prediction',
                marker=dict(color='red', size=12, symbol='square'),
                hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
            )
        )
        
        # Actual value if available
        if actual_debt_prediction_mask.any():
            rnn_debt_fig.add_trace(
                go.Scatter(
                    x=[prediction_year],
                    y=[actual_debt_ratio],
                    mode='markers',
                    name=f'Actual {prediction_year}',
                    marker=dict(color='green', size=14, symbol='diamond'),
                    hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
                )
            )
        
        # Add reference lines
        # rnn_debt_fig.add_hline(y=0.5, line_dash="dash", line_color="green",
        #                       annotation_text="Good Level (0.5)", annotation_position="top")
        # rnn_debt_fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
        #                       annotation_text="Moderate Level (0.7)", annotation_position="top")
        # rnn_debt_fig.add_hline(y=1.0, line_dash="dash", line_color="red",
        #                       annotation_text="High Risk Level (1.0)", annotation_position="top")
        
        # Update layout
        rnn_debt_fig.update_layout(
            title=f'RNN Model Debt Ratio Trends - Debt to Equity Ratio - {symbol}',
            xaxis_title='Year',
            yaxis_title='Debt Ratio',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Convert plot to HTML
        rnn_debt_ratio_metrics_html = rnn_debt_fig.to_html(full_html=False, include_plotlyjs='cdn')

    """
    Random Forest
    """
    """
    Random Forest
    """
    fig_rf_html = train_and_predict_ratio_random_forest(
        df_balance_sheet,
        symbol,
        target_col="currentRatio",
        prediction_year=2023,
        features=current_ratio_prediction,
    )
    qr_fig_rf_html = train_and_predict_ratio_random_forest(
        df_balance_sheet,
        symbol,
        target_col="quickRatio",
        prediction_year=2023,
        features=quick_ratio_prediction,
    )
    debt_ratio_fig_rf_html = train_and_predict_ratio_random_forest(
        df_balance_sheet,
        symbol,
        target_col="debtRatio",
        prediction_year=2023,
        features=debt_ratio_prediction,
    )
    # # Prepare data: use numeric columns + encoded symbol
    # df = df_balance_sheet[df_balance_sheet["ticker"] == symbol].copy()
    # numberic_cols = df_balance_sheet.select_dtypes(include=[np.number]).columns.tolist()
    # df[numberic_cols].replace([np.inf, -np.inf], 0)
    # le = LabelEncoder()
    # df['symbol_enc'] = le.fit_transform(df['ticker'].astype(str))
    #
    # # features: all numeric columns except target 'asset', plus encoded symbol
    # feature_cols = [c for c in numberic_cols if c != 'currentRatio'] + ['symbol_enc']
    # X = df[feature_cols].astype(float)
    # y = df['currentRatio'].astype(float)
    # y.fillna(0, inplace=True)
    #
    # # train / test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #
    # # train Random Forest
    # rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    # rf.fit(X_train, y_train)
    #
    # # feature importances
    # importances = rf.feature_importances_
    # feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    # print("Top features (feature, importance):")
    # for f, imp in feat_imp[:10]:
    #     print(f, round(imp, 4))
    #
    # # Plot trend for selected symbol (use existing symbol_selected if available)
    # sym_to_plot = symbol if (
    #             'symbol_selected' in globals() and symbol in df['ticker'].unique()) else (
    #     symbols[0] if 'symbols' in globals() and len(symbols) > 0 else df['ticker'].iloc[0])
    # df_sym = df[df['ticker'] == sym_to_plot].sort_values('year').reset_index(drop=True)
    #
    # # Prepare inputs for the symbol (use same feature columns)
    # X_sym = df_sym[feature_cols].astype(float)
    # pred_sym = rf.predict(X_sym)
    #
    # # Predict next year using last available row with year incremented
    # last = df_sym[df_sym['year'] == 2023].iloc[0]
    # next_row = last.copy()
    # next_row['year'] = int(last['year']) + 1
    # # keep other numeric features unchanged (this is a basic next-year estimate)
    # next_row['symbol_enc'] = le.transform([sym_to_plot])[0]
    # X_next = next_row[feature_cols].astype(float).values.reshape(1, -1)
    # pred_next = float(rf.predict(X_next)[0])
    #
    # # Plot with Plotly
    # fig_rf = go.Figure()
    # fig_rf.add_trace(go.Scatter(x=df_sym['year'], y=df_sym['currentRatio'],
    #                             mode='lines+markers', name=f'{sym_to_plot} History',
    #                             line=dict(width=2), marker=dict(size=6)))
    # fig_rf.add_trace(go.Scatter(x=df_sym['year'], y=pred_sym,
    #                             mode='lines+markers', name='Predicted (in-sample)',
    #                             line=dict(width=2, dash='dot'), marker=dict(size=6, symbol='circle-open')))
    # fig_rf.add_trace(go.Scatter(x=[next_row['year']], y=[pred_next],
    #                             mode='markers+text', name=f'Prediction {int(next_row["year"])}',
    #                             marker=dict(symbol='x', size=12, color='red'),
    #                             text=[f'{pred_next:,.0f}'], textposition='top center'))
    # fig_rf.update_layout(title=f'Random Forest: {sym_to_plot} current ratio (historical + predictions)',
    #                      xaxis_title='Year', yaxis_title='Asset', width=900, height=500)
    # fig_rf_html = fig_rf.to_html(full_html=False, include_plotlyjs='cdn')
    """
    End Random Forest
    """
    """
    End Random Forest
    """

    symbols = await get_symbols(session)
    context = {
        "request": request,
        "correlation_metrics_html": correlation_metrics_html,
        "current_ratio_metrics_html": current_ratio_metrics_html,
        "current_ratio_metrics_text": annotation_text,
        "rnn_current_ratio_metrics_html": rnn_current_ratio_metrics_html,
        "rnn_annotation_text": rnn_annotation_text,
        "quick_ratio_metrics_html": quick_ratio_metrics_html,
        "quick_ratio_metrics_text": quick_ratio_annotation_text,
        "rnn_quick_ratio_metrics_html": rnn_quick_ratio_metrics_html,
        "rnn_quick_ratio_annotation_text": rnn_quick_ratio_annotation_text,
        "debt_ratio_metrics_html": debt_ratio_metrics_html,
        "debt_ratio_metrics_text": debt_ratio_annotation_text,
        "rnn_debt_ratio_metrics_html": rnn_debt_ratio_metrics_html,
        "rnn_debt_ratio_annotation_text": rnn_debt_ratio_annotation_text,
        "fig_rf_html": fig_rf_html,
        "qr_fig_rf_html": qr_fig_rf_html,
        "debt_ratio_fig_rf_html": debt_ratio_fig_rf_html,
        "features_2_predict": feature_cols,
        "summary": summary,
        "symbols": symbols,
        "symbol": symbol,
        "model_type": model_type,
    }

    return templates.TemplateResponse("balancesheet.html", context=context)
