import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.ext.asyncio import AsyncSession

from core.helpers import get_lstm_model, get_rnn_model, get_cash_flow, get_symbols, create_sequences, get_balance_sheet, \
    train_and_predict_ratio_linear_regression, train_and_predict_ratio_random_forest
from db import session_manager


router = APIRouter(
    prefix="/cashflow",
    tags=["income-statement-dashboard"]
)
templates = Jinja2Templates(directory="templates")
FREE_CASH_FLOW_TO_EQUITY_BRIEF = """
FCFE represents cash remaining for equity holders after all operational and debt obligations. 
High FCFE suggests strong financial health and the capacity to reward shareholders without 
relying on external financing.<br><br>
"""
FREE_CASH_FLOW_BRIEF = """
Free Cash Flow (FCF) indicates a company's capacity to handle debt and reward shareholders.<br><br>
"""


@router.get("/", response_class=HTMLResponse)
async def cashflow_dashboard(
        request: Request, session: AsyncSession = Depends(session_manager.session),
        symbol: str = Query('FPT', description="Stock symbol"),
        prediction_year: int = Query(2023, description="Year to predict"),
        model_type: str = Query('LSTM', description="Model to use for prediction"),
        yearly: bool = Query(True, description="Use yearly data"),
):
    """
    Generate heatmap chart showing correlation matrix between balance sheet features
    """
    symbol = symbol.upper()
    look_back = 3

    # Fetch data from database
    try:
        df_cashflow = await get_cash_flow(session, symbol, prediction_year, yearly)
        df_balance_sheet = await get_balance_sheet(session, symbol, prediction_year, yearly)
        df_cashflow["equity"] =  df_balance_sheet["equity"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

    feature_cols = [
        'investCost',
        'fromInvest',
        'fromFinancial',
        'fromSale',
        'freeCashFlow',
    ]

    df_cashflow = df_cashflow.sort_values(['ticker', 'year']).reset_index(drop=True)
    df_cashflow = df_cashflow.fillna(value=0)

    # Calculate summary statistics
    summary = {}

    # Merge balance sheet equity with cashflow for FCFE calculation
    df_cashflow['fcfe'] = df_cashflow.get('fromSale', pd.Series(0)) - df_cashflow.get('investCost', pd.Series(0)) + df_cashflow['equity']

    if len(df_cashflow) > 0:
        summary['latest_fcf'] = float(df_cashflow['freeCashFlow'].iloc[-1]) if 'freeCashFlow' in df_cashflow.columns else 0
        summary['latest_fcfe'] = float(df_cashflow['fcfe'].iloc[-1]) if 'fcfe' in df_cashflow.columns else 0

    # Gross Profit Margin
    df_cashflow["fcfe"] = df_cashflow["fromSale"] - df_cashflow["investCost"] + df_cashflow["equity"]

    # Calculate correlation matrix for all balance sheet features
    # Filter out non-numeric columns and ensure feature columns exist
    numeric_cols = [col for col in feature_cols if col in df_cashflow.columns]

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
            title='Cash Flow Correlation Heatmap - Insufficient Data',
            height=600,
            width=1600,
        )
        plot_html = fig.to_html(full_html=False)
        return templates.TemplateResponse("cashflow.html", {"request": request, "plot_html": plot_html})

    # Calculate correlation matrix
    corr_matrix = df_cashflow[numeric_cols].corr()

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

    # Filter for selected symbol
    df_symbol = df_cashflow[df_cashflow['ticker'] == symbol].copy()

    # Check data sufficiency
    if len(df_symbol) < look_back + 2:
        raise HTTPException(status_code=400,
                            detail=f"Insufficient data for {symbol}. Need at least {look_back + 2} years.")

    # Prepare data
    predicted_features = ['fromSale', 'investCost', 'freeCashFlow', 'equity']
    df_symbol = df_symbol.sort_values('year')
    df_symbol = df_symbol.dropna(subset=predicted_features).reset_index(drop=True)

    if len(df_symbol) < look_back + 2:
        raise HTTPException(status_code=400,
                            detail=f"Insufficient valid data after removing NaN values.")

    data = df_symbol[predicted_features].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Get data before prediction year
    df_before_prediction = df_symbol[df_symbol['year'] < prediction_year].copy()
    data_before_prediction = df_before_prediction[predicted_features].values
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
        combined_data = np.vstack([data[-look_back:], df_symbol[predicted_features].values])
        scaled_combined = scaler.transform(combined_data)

        x_val, y_val = create_sequences(scaled_combined, look_back)

        if len(x_val) > 0:
            val_x.append(x_val)
            val_y.append(y_val)

    lstm_model_path = f"models/cashflow_{symbol}.keras"
    lstm_model = get_lstm_model(
        lstm_model_path,
        len(df_cashflow),
        predicted_features,
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
        pred_input = last_sequence.reshape(1, look_back, len(predicted_features))
        pred = lstm_model.predict(pred_input, verbose=0)
        future_predictions.append(pred[0])
        last_sequence = np.vstack([last_sequence[1:], pred[0]])

    # Inverse transform predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions))

    # Calculate predicted current ratio
    # predicted_features = ['fromSale', 'investCost', 'freeCashFlow', 'equity']
    predicted_values = future_predictions[-1]
    predicted_from_sale, predicted_invest_cost = predicted_values[0], predicted_values[1]
    predicted_fcf, predicted_equity = predicted_values[2], predicted_values[3]

    predicted_fcfe = predicted_from_sale - predicted_invest_cost + predicted_equity

    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Cashflow Features Correlation Heatmap<br><sub>Mean: '
                    f'{mean_corr:.3f}, Median: {median_corr:.3f}, Std: {std_corr:.3f}</sub><b>',
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

    # Build annotation
    # FCFE (Free Cash Flow to Equity) annotation explanation
    fcfe_metrics_text = f"<b>About Free Cash Flow to Equity (FCFE)</b><br>"
    fcfe_metrics_text += FREE_CASH_FLOW_TO_EQUITY_BRIEF
    fcfe_metrics_text += f"<b>{prediction_year} Current FCFE for {symbol}</b><br>"
    fcfe_metrics_text += f"<b>Predicted FCFE</b>: {predicted_fcfe:.2f}<br><br>"
    fcfe_metrics_text += f"<b>Predicted Components:</b><br>"
    fcfe_metrics_text += f"• Predicted Cash From Sale: {predicted_from_sale:,.2f}B VND<br>"
    fcfe_metrics_text += f"• Predicted Invest Cost: {predicted_invest_cost:,.2f}B VND<br><br>"
    fcfe_metrics_text += f"• Predicted Equity: {predicted_equity:,.2f}B VND<br><br>"

    # Check if actual data exists
    actual_prediction_mask = df_symbol['year'] == prediction_year
    actual_fcfe = None

    if actual_prediction_mask.any():
        actual_fcfe = df_symbol[actual_prediction_mask]['fcfe'].values[0]
        difference = predicted_fcf - actual_fcfe
        percentage_diff = (difference / actual_fcfe) * 100 if actual_fcfe != 0 else 0

        fcfe_metrics_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        fcfe_metrics_text += f"• Actual {prediction_year}: {actual_fcfe:.2f}<br>"
        fcfe_metrics_text += f"• Difference: {difference:.2f} ({percentage_diff:+.2f}%)<br><br>"

    # Add FCFE assessment based on historical data and predicted value
    historical_fcfe = df_symbol['fcfe'].values
    historical_fcfe_positive = historical_fcfe[historical_fcfe > 0]
    
    if len(historical_fcfe_positive) > 0:
        avg_historical_fcfe = np.mean(historical_fcfe_positive)
        median_historical_fcfe = np.median(historical_fcfe_positive)
    else:
        avg_historical_fcfe = 0
        median_historical_fcfe = 0
    
    fcfe_metrics_text += f"<b>FCFE Financial Health Assessment:</b><br>"
    
    # Assess based on positive/negative and comparison with historical data
    if predicted_fcfe > 0:
        if len(historical_fcfe_positive) > 0:
            if predicted_fcfe >= avg_historical_fcfe * 1.2:
                fcfe_metrics_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
                fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) significantly exceeds historical average ({avg_historical_fcfe:,.2f}B VND)<br>"
                fcfe_metrics_text += f"• Strong ability to self-fund dividends and share repurchases<br>"
                fcfe_metrics_text += f"• Company demonstrates robust cash generation for equity holders<br>"
            elif predicted_fcfe >= avg_historical_fcfe * 0.8:
                fcfe_metrics_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
                fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) aligns with historical performance ({avg_historical_fcfe:,.2f}B VND)<br>"
                fcfe_metrics_text += f"• Company can comfortably fund shareholder returns without external financing<br>"
                fcfe_metrics_text += f"• Healthy cash flow generation for equity stakeholders<br>"
            else:
                fcfe_metrics_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
                fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) is below historical average ({avg_historical_fcfe:,.2f}B VND)<br>"
                fcfe_metrics_text += f"• Company can still fund dividends but with reduced capacity<br>"
                fcfe_metrics_text += f"• Monitor cash flow trends and capital allocation decisions<br>"
        else:
            # No historical positive FCFE data, but current is positive
            fcfe_metrics_text += f"• Expected Status: <b style='color:blue'>Positive Turnaround</b><br>"
            fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) is positive, indicating improved cash generation<br>"
            fcfe_metrics_text += f"• Company shows potential to generate cash for equity holders<br>"
            fcfe_metrics_text += f"• Monitor sustainability of positive cash flow trend<br>"
    else:
        # Negative FCFE
        if len(historical_fcfe_positive) > 0:
            fcfe_metrics_text += f"• Expected Status: <b style='color:red'>Concerning</b><br>"
            fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) is negative, below historical positive average ({avg_historical_fcfe:,.2f}B VND)<br>"
            fcfe_metrics_text += f"• Company may require external financing to fund dividends or share repurchases<br>"
            fcfe_metrics_text += f"• Evaluate capital structure and cash flow management strategies<br>"
        else:
            fcfe_metrics_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            fcfe_metrics_text += f"• Predicted FCFE ({predicted_fcfe:,.2f}B VND) is negative<br>"
            fcfe_metrics_text += f"• Company lacks sufficient cash flow to fund equity shareholder returns<br>"
            fcfe_metrics_text += f"• May need external financing or debt restructuring to maintain operations<br>"

    # Create visualization
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['fcfe'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current FCFE: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    fig.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_fcfe],
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
                y=[actual_fcfe],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Update layout
    fig.update_layout(
        title=f'LSTM Free Cash Flow to Equity Indicator - '
              f'Ability of an enterprise which generates cash flow for stakeholders. - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current FCFE',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    fcfe_metrics_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Build FCF annotation
    # FCF (Free Cash Flow) annotation explanation
    fcf_metrics_text = f"<b>About Free Cash Flow (FCF)</b><br>"
    fcf_metrics_text += FREE_CASH_FLOW_BRIEF
    fcf_metrics_text += f"<b>{prediction_year} Free Cash Flow for {symbol}</b><br>"
    fcf_metrics_text += f"<b>Predicted FCF</b>: {predicted_fcf:,.2f}B VND<br><br>"
    
    # Check if actual FCF data exists
    actual_fcf = None
    if actual_prediction_mask.any():
        actual_fcf = df_symbol[actual_prediction_mask]['freeCashFlow'].values[0]
        difference_fcf = predicted_fcf - actual_fcf
        percentage_diff_fcf = (difference_fcf / actual_fcf) * 100 if actual_fcf != 0 else 0
        
        fcf_metrics_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        fcf_metrics_text += f"• Actual {prediction_year}: {actual_fcf:,.2f}B VND<br>"
        fcf_metrics_text += f"• Difference: {difference_fcf:,.2f}B VND ({percentage_diff_fcf:+.2f}%)<br><br>"
    
    # Add FCF assessment based on historical data and predicted value
    historical_fcf = df_symbol['freeCashFlow'].values
    historical_fcf_positive = historical_fcf[historical_fcf > 0]
    
    if len(historical_fcf_positive) > 0:
        avg_historical_fcf = np.mean(historical_fcf_positive)
        median_historical_fcf = np.median(historical_fcf_positive)
    else:
        avg_historical_fcf = 0
        median_historical_fcf = 0
    
    fcf_metrics_text += f"<b>FCF Financial Health Assessment:</b><br>"
    
    # Assess based on positive/negative and comparison with historical data
    if predicted_fcf > 0:
        if len(historical_fcf_positive) > 0:
            if predicted_fcf >= avg_historical_fcf * 1.2:
                fcf_metrics_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
                fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) significantly exceeds historical average ({avg_historical_fcf:,.2f}B VND)<br>"
                fcf_metrics_text += f"• Strong cash generation capability for all business activities<br>"
                fcf_metrics_text += f"• Company demonstrates robust ability to fund operations and growth<br>"
            elif predicted_fcf >= avg_historical_fcf * 0.8:
                fcf_metrics_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
                fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) aligns with historical performance ({avg_historical_fcf:,.2f}B VND)<br>"
                fcf_metrics_text += f"• Company can comfortably fund operations, dividends, and debt obligations<br>"
                fcf_metrics_text += f"• Healthy cash flow generation for business sustainability<br>"
            else:
                fcf_metrics_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
                fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) is below historical average ({avg_historical_fcf:,.2f}B VND)<br>"
                fcf_metrics_text += f"• Company can still fund operations but with reduced cash buffer<br>"
                fcf_metrics_text += f"• Monitor cash flow trends and operational efficiency<br>"
        else:
            # No historical positive FCF data, but current is positive
            fcf_metrics_text += f"• Expected Status: <b style='color:blue'>Positive Turnaround</b><br>"
            fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) is positive, indicating improved cash generation<br>"
            fcf_metrics_text += f"• Company shows potential to generate cash for business activities<br>"
            fcf_metrics_text += f"• Monitor sustainability of positive cash flow trend<br>"
    else:
        # Negative FCF
        if len(historical_fcf_positive) > 0:
            fcf_metrics_text += f"• Expected Status: <b style='color:red'>Concerning</b><br>"
            fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) is negative, below historical positive average ({avg_historical_fcf:,.2f}B VND)<br>"
            fcf_metrics_text += f"• Company may require external financing to fund operations and obligations<br>"
            fcf_metrics_text += f"• Evaluate operational efficiency and cash flow management strategies<br>"
        else:
            fcf_metrics_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            fcf_metrics_text += f"• Predicted FCF ({predicted_fcf:,.2f}B VND) is negative<br>"
            fcf_metrics_text += f"• Company lacks sufficient cash flow to fund business activities<br>"
            fcf_metrics_text += f"• May need external financing or operational restructuring to maintain operations<br>"

    # Create visualization
    fcf_chart = go.Figure()
    # Historical data
    fcf_chart.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['freeCashFlow'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current FCF: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    fcf_chart.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_fcf],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=12, symbol='square'),
            hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        )
    )

    # Actual value if available
    if actual_fcf is not None:
        fcf_chart.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[actual_fcf],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Update layout
    fcf_chart.update_layout(
        title=f'LSTM Free Cash Flow Indicator - '
              f'Ability of enterprise to creat cash flow for all business activities - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Free Cash Flow',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    fcf_metrics_html = fcf_chart.to_html(full_html=False, include_plotlyjs='cdn')

    """
    Linear Regression Predictions
    """
    if prediction_year < 2025:
        lr_fcf_html = train_and_predict_ratio_linear_regression(
            df_cashflow, symbol, "freeCashFlow", prediction_year, "Free Cash Flow"
        )

        lr_fcfe_html = train_and_predict_ratio_linear_regression(
            df_cashflow, symbol, "fcfe", prediction_year, "Free Cash Flow to Equity"
        )
    else:
        lr_fcf_html = None
        lr_fcfe_html = None
    """
    End Linear Regression
    """

    """
    Random Forest
    """
    if prediction_year < 2025:
        fig_fcf_html = train_and_predict_ratio_random_forest(
            df_cashflow,
            symbol,
            target_col="freeCashFlow",
            prediction_year=2023,
            features=["freeCashFlow"],
        )
        fig_fcfe_html = train_and_predict_ratio_random_forest(
            df_cashflow,
            symbol,
            target_col="fcfe",
            prediction_year=2023,
            features=["fcfe", "fromSale", "investCost", "equity"],
        )
    else:
        fig_fcf_html = None
        fig_fcfe_html = None
    """
    End Random Forest
    """

    """RNN Prediction"""
    """
        RNN Prediction
        """
    rnn_model_path = f"models/rnn_cashflow_{symbol}.keras"
    rnn_model = get_rnn_model(
        rnn_model_path,
        len(df_cashflow),
        predicted_features,
        x_train=x_train,
        y_train=y_train,
        val_x=val_x,
        val_y=val_y,
    )

    # Make RNN prediction
    last_sequence_rnn = scaled_data_before_prediction[-look_back:].copy()
    rnn_future_predictions = []

    for _ in range(num_years_to_predict):
        pred_input_rnn = last_sequence_rnn.reshape(1, look_back, len(predicted_features))
        pred_rnn = rnn_model.predict(pred_input_rnn, verbose=0)
        rnn_future_predictions.append(pred_rnn[0])
        last_sequence_rnn = np.vstack([last_sequence_rnn[1:], pred_rnn[0]])

    # Inverse transform predictions
    rnn_future_predictions = scaler.inverse_transform(np.array(rnn_future_predictions))

    # Calculate RNN predicted values
    rnn_predicted_values = rnn_future_predictions[-1]
    rnn_predicted_from_sale, rnn_predicted_invest_cost = rnn_predicted_values[0], rnn_predicted_values[1]
    rnn_predicted_fcf, rnn_predicted_equity = rnn_predicted_values[2], rnn_predicted_values[3]

    rnn_predicted_fcfe = rnn_predicted_from_sale - rnn_predicted_invest_cost + rnn_predicted_equity

    # --- RNN FCFE Text Generation ---
    rnn_fcfe_metrics_text = f"<b>About Free Cash Flow to Equity (FCFE)</b><br>"
    rnn_fcfe_metrics_text += FREE_CASH_FLOW_TO_EQUITY_BRIEF
    rnn_fcfe_metrics_text += f"<b>{prediction_year} Current FCFE for {symbol}</b><br>"
    rnn_fcfe_metrics_text += f"<b>Predicted FCFE</b>: {rnn_predicted_fcfe:.2f}<br><br>"
    rnn_fcfe_metrics_text += f"<b>Predicted Components:</b><br>"
    rnn_fcfe_metrics_text += f"• Predicted Cash From Sale: {rnn_predicted_from_sale:,.2f}B VND<br>"
    rnn_fcfe_metrics_text += f"• Predicted Invest Cost: {rnn_predicted_invest_cost:,.2f}B VND<br><br>"
    rnn_fcfe_metrics_text += f"• Predicted Equity: {rnn_predicted_equity:,.2f}B VND<br><br>"

    if actual_prediction_mask.any():
        actual_fcfe = df_symbol[actual_prediction_mask]['fcfe'].values[0]
        difference_rnn = rnn_predicted_fcfe - actual_fcfe
        percentage_diff_rnn = (difference_rnn / actual_fcfe) * 100 if actual_fcfe != 0 else 0

        rnn_fcfe_metrics_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        rnn_fcfe_metrics_text += f"• Actual {prediction_year}: {actual_fcfe:.2f}<br>"
        rnn_fcfe_metrics_text += f"• Difference: {difference_rnn:.2f} ({percentage_diff_rnn:+.2f}%)<br><br>"

    rnn_fcfe_metrics_text += f"<b>FCFE Financial Health Assessment:</b><br>"
    if rnn_predicted_fcfe > 0:
        if len(historical_fcfe_positive) > 0:
            if rnn_predicted_fcfe >= avg_historical_fcfe * 1.2:
                rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
                rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) significantly exceeds historical average<br>"
            elif rnn_predicted_fcfe >= avg_historical_fcfe * 0.8:
                rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
                rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) aligns with historical performance<br>"
            else:
                rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
                rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) is below historical average<br>"
        else:
            rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:blue'>Positive Turnaround</b><br>"
            rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) is positive<br>"
    else:
        if len(historical_fcfe_positive) > 0:
            rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:red'>Concerning</b><br>"
            rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) is negative<br>"
        else:
            rnn_fcfe_metrics_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            rnn_fcfe_metrics_text += f"• Predicted FCFE ({rnn_predicted_fcfe:,.2f}B VND) is negative<br>"

    # --- RNN FCFE Chart ---
    rnn_fcfe_fig = go.Figure()
    rnn_fcfe_fig.add_trace(go.Scatter(x=df_symbol['year'], y=df_symbol['fcfe'], mode='lines+markers', name='Historical',
                                      line=dict(color='blue', width=2), marker=dict(size=8),
                                      hovertemplate='Year: %{x}<br>Current FCFE: %{y:.2f}<extra></extra>'))
    rnn_fcfe_fig.add_trace(go.Scatter(x=[prediction_year], y=[rnn_predicted_fcfe], mode='markers', name='Prediction',
                                      marker=dict(color='red', size=12, symbol='square'),
                                      hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'))
    if actual_prediction_mask.any():
        rnn_fcfe_fig.add_trace(
            go.Scatter(x=[prediction_year], y=[actual_fcfe], mode='markers', name=f'Actual {prediction_year}',
                       marker=dict(color='green', size=14, symbol='diamond'),
                       hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'))
    rnn_fcfe_fig.update_layout(title=f'RNN Free Cash Flow to Equity Indicator - {symbol}', xaxis_title='Year',
                               yaxis_title='Current FCFE', height=600, showlegend=True, hovermode='x unified',
                               template='plotly_white')
    rnn_fcfe_metrics_html = rnn_fcfe_fig.to_html(full_html=False, include_plotlyjs='cdn')

    # --- RNN FCF Text Generation ---
    rnn_fcf_metrics_text = f"<b>About Free Cash Flow (FCF)</b><br>"
    rnn_fcf_metrics_text += FREE_CASH_FLOW_BRIEF
    rnn_fcf_metrics_text += f"<b>{prediction_year} Free Cash Flow for {symbol}</b><br>"
    rnn_fcf_metrics_text += f"<b>Predicted FCF</b>: {rnn_predicted_fcf:,.2f}B VND<br><br>"

    if actual_fcf is not None:
        difference_rnn_fcf = rnn_predicted_fcf - actual_fcf
        percentage_diff_rnn_fcf = (difference_rnn_fcf / actual_fcf) * 100 if actual_fcf != 0 else 0
        rnn_fcf_metrics_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        rnn_fcf_metrics_text += f"• Actual {prediction_year}: {actual_fcf:,.2f}B VND<br>"
        rnn_fcf_metrics_text += f"• Difference: {difference_rnn_fcf:,.2f}B VND ({percentage_diff_rnn_fcf:+.2f}%)<br><br>"

    rnn_fcf_metrics_text += f"<b>FCF Financial Health Assessment:</b><br>"
    if rnn_predicted_fcf > 0:
        if len(historical_fcf_positive) > 0:
            if rnn_predicted_fcf >= avg_historical_fcf * 1.2:
                rnn_fcf_metrics_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
                rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) significantly exceeds historical average<br>"
            elif rnn_predicted_fcf >= avg_historical_fcf * 0.8:
                rnn_fcf_metrics_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
                rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) aligns with historical performance<br>"
            else:
                rnn_fcf_metrics_text += f"• Expected Status: <b style='color:orange'>Moderate</b><br>"
                rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) is below historical average<br>"
        else:
            rnn_fcf_metrics_text += f"• Expected Status: <b style='color:blue'>Positive Turnaround</b><br>"
            rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) is positive<br>"
    else:
        if len(historical_fcf_positive) > 0:
            rnn_fcf_metrics_text += f"• Expected Status: <b style='color:red'>Concerning</b><br>"
            rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) is negative<br>"
        else:
            rnn_fcf_metrics_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
            rnn_fcf_metrics_text += f"• Predicted FCF ({rnn_predicted_fcf:,.2f}B VND) is negative<br>"

    # --- RNN FCF Chart ---
    rnn_fcf_chart = go.Figure()
    rnn_fcf_chart.add_trace(
        go.Scatter(x=df_symbol['year'], y=df_symbol['freeCashFlow'], mode='lines+markers', name='Historical',
                   line=dict(color='blue', width=2), marker=dict(size=8),
                   hovertemplate='Year: %{x}<br>Current FCF: %{y:.2f}<extra></extra>'))
    rnn_fcf_chart.add_trace(go.Scatter(x=[prediction_year], y=[rnn_predicted_fcf], mode='markers', name='Prediction',
                                       marker=dict(color='red', size=12, symbol='square'),
                                       hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'))
    if actual_fcf is not None:
        rnn_fcf_chart.add_trace(
            go.Scatter(x=[prediction_year], y=[actual_fcf], mode='markers', name=f'Actual {prediction_year}',
                       marker=dict(color='green', size=14, symbol='diamond'),
                       hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'))
    rnn_fcf_chart.update_layout(title=f'RNN Free Cash Flow Indicator - {symbol}', xaxis_title='Year',
                                yaxis_title='Current Free Cash Flow', height=600, showlegend=True,
                                hovermode='x unified', template='plotly_white')
    rnn_fcf_metrics_html = rnn_fcf_chart.to_html(full_html=False, include_plotlyjs='cdn')
    """End RNN Prediction"""

    symbols = await get_symbols(session)
    context = {
        "request": request,
        "correlation_metrics_html": correlation_metrics_html,
        "fcfe_metrics_html": fcfe_metrics_html,
        "fcfe_metrics_text": fcfe_metrics_text,
        "fcf_metrics_html": fcf_metrics_html,
        "fcf_metrics_text": fcf_metrics_text,
        "summary": summary,
        "symbols": symbols,
        "symbol": symbol,
        "model_type": model_type,
        # Linear Regression
        "lr_fcf_html": lr_fcf_html,
        "lr_fcfe_html": lr_fcfe_html,
        # Random Forest
        "fig_fcf_html": fig_fcf_html,
        "fig_fcfe_html": fig_fcfe_html,
        # RNN
        "rnn_fcfe_metrics_html": rnn_fcfe_metrics_html,
        "rnn_fcfe_metrics_text": rnn_fcfe_metrics_text,
        "rnn_fcf_metrics_html": rnn_fcf_metrics_html,
        "rnn_fcf_metrics_text": rnn_fcf_metrics_text,
    }

    return templates.TemplateResponse("cashflow.html", context=context)
