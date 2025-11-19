import os

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, LeakyReLU, BatchNormalization, SimpleRNN
from tensorflow.keras.models import Sequential, load_model

from models.balancesheet import BalanceSheet
from models.income_statement import IncomeStatement
from models.stock import Stock
from models.cashflow import Cashflow


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


async def get_balance_sheet(session: AsyncSession, symbol: str, year: int, yearly: bool = True):
    stmt_bs = (select(BalanceSheet)
               .where(BalanceSheet.yearly == True))
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
