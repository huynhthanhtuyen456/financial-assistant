import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import APIRouter, Query, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy.ext.asyncio import AsyncSession

from core.helpers import get_lstm_model, get_rnn_model, get_income_statement, get_symbols, create_sequences
from db import session_manager

router = APIRouter(
    prefix="/income-statement",
    tags=["income-statement-dashboard"]
)
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def income_statement(
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
    look_back = 3

    # Fetch data from database
    try:
        df_income_statement = await get_income_statement(session, symbol, prediction_year, yearly)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

    feature_cols = [
        'revenue',
        'yearRevenueGrowth',
        'quarterRevenueGrowth',
        'costOfGoodSold',
        'grossProfit',
        'operationExpense',
        'operationProfit',
        'yearOperationProfitGrowth',
        'quarterOperationProfitGrowth',
        'interestExpense',
        'preTaxProfit',
        'postTaxProfit',
        'shareHolderIncome',
        'yearShareHolderIncomeGrowth',
        'quarterShareHolderIncomeGrowth',
        'investProfit',
        'serviceProfit',
        'otherProfit',
        'provisionExpense',
        'operationIncome',
        'ebitda',
    ]

    df_income_statement = df_income_statement.sort_values(['ticker', 'year']).reset_index(drop=True)
    df_income_statement = df_income_statement.fillna(value=0)

    # Gross Profit Margin
    df_income_statement["gross_profit_margin"] = ((df_income_statement["grossProfit"] / df_income_statement["revenue"])
                                                  * 100)
    df_income_statement["operating_profit_margin"] = ((df_income_statement["preTaxProfit"]
                                                       / df_income_statement["revenue"])) * 100
    df_income_statement["net_profit_margin"] = ((df_income_statement["postTaxProfit"]
                                                 / df_income_statement["revenue"]) * 100)

    # Process each company separately and calculate correlations
    symbols = df_income_statement['ticker'].unique()

    # Calculate correlation matrix for all balance sheet features
    # Filter out non-numeric columns and ensure feature columns exist
    numeric_cols = [col for col in feature_cols if col in df_income_statement.columns]

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
        return templates.TemplateResponse("income_statement.html", {"request": request, "plot_html": plot_html})

    # Calculate correlation matrix
    corr_matrix = df_income_statement[numeric_cols].corr()

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
    df_symbol = df_income_statement[df_income_statement['ticker'] == symbol].copy()

    # Check data sufficiency
    if len(df_symbol) < look_back + 2:
        raise HTTPException(status_code=400,
                            detail=f"Insufficient data for {symbol}. Need at least {look_back + 2} years.")

    # Prepare data
    predicted_features = ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']
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

    lstm_model_path = f"models/income_statement_{symbol}.keras"
    lstm_model = get_lstm_model(
        lstm_model_path,
        len(df_income_statement),
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
    # predicted_features = ['grossProfit', 'revenue', 'preTaxProfit', 'postTaxProfit']
    predicted_values = future_predictions[-1]
    predicted_gross_profit, predicted_revenue = predicted_values[0], predicted_values[1]
    predicted_pre_tax_profit, predicted_post_tax_profit = predicted_values[2], predicted_values[3]

    predicted_gross_profit_margin = (predicted_gross_profit / predicted_revenue) * 100
    predicted_operating_profit_margin = (predicted_pre_tax_profit / predicted_revenue) * 100
    predicted_net_profit_margin = (predicted_post_tax_profit / predicted_revenue) * 100

    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Income Statement Features Correlation Heatmap<br><sub>Mean: '
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
    gross_profit_margin_metrics_text = f"<b>{prediction_year} Current Ratio Prediction for {symbol}</b><br>"
    gross_profit_margin_metrics_text += f"<b>Predicted Gross Profit Margin</b>: {predicted_gross_profit_margin:.2f}<br><br>"
    gross_profit_margin_metrics_text += f"<b>Predicted Components:</b><br>"
    gross_profit_margin_metrics_text += f"• Predicted Pre Tax Profit: {predicted_pre_tax_profit:,.2f}B VND<br>"
    gross_profit_margin_metrics_text += f"• Predicted Revenue: {predicted_revenue:,.2f}B VND<br><br>"

    # Check if actual data exists
    actual_prediction_mask = df_symbol['year'] == prediction_year
    actual_gross_profit_margin = None

    if actual_prediction_mask.any():
        actual_gross_profit_margin = df_symbol[actual_prediction_mask]['gross_profit_margin'].values[0]
        difference = predicted_gross_profit_margin - actual_gross_profit_margin
        percentage_diff = (difference / actual_gross_profit_margin) * 100 if actual_gross_profit_margin != 0 else 0

        gross_profit_margin_metrics_text += f"<b>Comparison with Actual {prediction_year}:</b><br>"
        gross_profit_margin_metrics_text += f"• Actual {prediction_year}: {actual_gross_profit_margin:.2f}<br>"
        gross_profit_margin_metrics_text += f"• Difference: {difference:.2f} ({percentage_diff:+.2f}%)<br><br>"

    # Add liquidity assessment
    gross_profit_margin_metrics_text += f"<b>Predicted Liquidity Assessment:</b><br>"
    if predicted_gross_profit_margin >= 2.0:
        gross_profit_margin_metrics_text += f"• Expected Status: <b style='color:green'>Excellent</b><br>"
        gross_profit_margin_metrics_text += f"• Company has strong ability to cover short-term obligations<br>"
        gross_profit_margin_metrics_text += f"• Current assets are {predicted_gross_profit_margin:.2f}x current liabilities<br>"
    elif predicted_gross_profit_margin >= 1.5:
        gross_profit_margin_metrics_text += f"• Expected Status: <b style='color:blue'>Good</b><br>"
        gross_profit_margin_metrics_text += f"• Company can comfortably meet short-term obligations<br>"
        gross_profit_margin_metrics_text += f"• Healthy liquidity position maintained<br>"
    elif predicted_gross_profit_margin >= 1.0:
        gross_profit_margin_metrics_text += f"• Expected Status: <b style='color:orange'>Adequate</b><br>"
        gross_profit_margin_metrics_text += f"• Company can meet obligations but with limited buffer<br>"
        gross_profit_margin_metrics_text += f"• Monitor closely for liquidity issues<br>"
    else:
        gross_profit_margin_metrics_text += f"• Expected Status: <b style='color:red'>Warning</b><br>"
        gross_profit_margin_metrics_text += f"• Current assets insufficient to cover current liabilities<br>"
        gross_profit_margin_metrics_text += f"• Potential liquidity concerns - ratio below 1.0<br>"

    # Create visualization
    fig = go.Figure()

    # Historical data
    fig.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['gross_profit_margin'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current Gross Profit Margin: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    fig.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_gross_profit_margin],
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
                y=[actual_gross_profit_margin],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Update layout
    fig.update_layout(
        title=f'LSTM Gross Profit Margin Indicator - '
              f'The money a company makes after accounting for its business costs - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Gross Profit Margin',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    gross_profit_margin_metrics_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Create visualization
    net_profit_margin_chart = go.Figure()
    actual_net_profit_margin = df_symbol[actual_prediction_mask]['net_profit_margin'].values[0]
    # Historical data
    net_profit_margin_chart.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['net_profit_margin'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current Net Profit Margin: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    net_profit_margin_chart.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_net_profit_margin],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=12, symbol='square'),
            hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        )
    )

    # Actual value if available
    if actual_prediction_mask.any():
        net_profit_margin_chart.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[actual_net_profit_margin],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Update layout
    net_profit_margin_chart.update_layout(
        title=f'LSTM Net Profit Margin Indicator - '
              f'How much profit (or net income) a business generates - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Net Profit Margin',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    net_profit_margin_metrics_html = net_profit_margin_chart.to_html(full_html=False, include_plotlyjs='cdn')

    # Create visualization
    operating_profit_margin_chart = go.Figure()
    actual_operating_profit_margin = df_symbol[actual_prediction_mask]['operating_profit_margin'].values[0]
    # Historical data
    operating_profit_margin_chart.add_trace(
        go.Scatter(
            x=df_symbol['year'],
            y=df_symbol['operating_profit_margin'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Current Operating Profit Margin: %{y:.2f}<extra></extra>'
        )
    )

    # Prediction
    operating_profit_margin_chart.add_trace(
        go.Scatter(
            x=[prediction_year],
            y=[predicted_operating_profit_margin],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=12, symbol='square'),
            hovertemplate='Year: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
        )
    )

    # Actual value if available
    if actual_prediction_mask.any():
        operating_profit_margin_chart.add_trace(
            go.Scatter(
                x=[prediction_year],
                y=[actual_operating_profit_margin],
                mode='markers',
                name=f'Actual {prediction_year}',
                marker=dict(color='green', size=14, symbol='diamond'),
                hovertemplate='Year: %{x}<br>Actual: %{y:.2f}<extra></extra>'
            )
        )

    # Update layout
    operating_profit_margin_chart.update_layout(
        title=f'LSTM Operating Profit Margin Indicator - '
              f'Pre tax profit created on every VND revenue - {symbol}',
        xaxis_title='Year',
        yaxis_title='Current Net Profit Margin',
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    # Convert plot to HTML
    operating_profit_margin_metrics_html = operating_profit_margin_chart.to_html(full_html=False, include_plotlyjs='cdn')

    symbols = await get_symbols(session)
    context = {
        "request": request,
        "correlation_metrics_html": correlation_metrics_html,
        "gross_profit_margin_metrics_html": gross_profit_margin_metrics_html,
        "gross_profit_margin_metrics_text": gross_profit_margin_metrics_text,
        "net_profit_margin_metrics_html": net_profit_margin_metrics_html,
        "operating_profit_margin_metrics_html": operating_profit_margin_metrics_html,
        "symbols": symbols,
        "symbol": symbol,
    }

    return templates.TemplateResponse("income_statement.html", context=context)
