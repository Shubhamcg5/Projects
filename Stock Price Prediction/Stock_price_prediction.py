# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 22:36:39 2025

@author: shubham
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import streamlit as st
import io
import time

st.title("Live Stock/Index Prediction with Buy/Sell Signals")

# Stock/Index Selection
def fetch_tickers():
    return {
        'Bank Nifty': '^NSEBANK',
        'Nifty 50': '^NSEI',
        'Sensex': '^BSESN',
        'Apple': 'AAPL',
        'Google': 'GOOGL',
        'Microsoft': 'MSFT'
    }

tickers = fetch_tickers()
selected_ticker_name = st.sidebar.selectbox("Select Stock/Index", list(tickers.keys()))
selected_ticker = tickers[selected_ticker_name]

# Fetch Data (Up to Today)
@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        data = yf.download(ticker, start='2019-01-01')
        if data.empty:
            raise ValueError("No data available")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

data = fetch_data(selected_ticker)

# Check if data is empty
if data.empty:
    st.error("Error: No data available for the selected ticker.")
    st.stop()

# Display Latest Data
st.subheader(f"Latest Data for {selected_ticker_name}")
st.dataframe(data.tail())

# Plot Closing Prices
st.subheader(f"Closing Price for {selected_ticker_name} (2019-Today)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['Close'], label=f'{selected_ticker_name} Close Price')
ax.set_title(f'{selected_ticker_name} Closing Price (2019-Today)')
ax.legend()
st.pyplot(fig)

# Data Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']])

# Create Sequences for GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

sequence_length = 30
X, y = create_sequences(scaled_data, sequence_length)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for GRU Input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Learning Rate Adjustment
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Build the Optimized GRU Model
model = Sequential()
model.add(Bidirectional(GRU(256, return_sequences=True, input_shape=(X_train.shape[1], 1), implementation=2)))
model.add(LayerNormalization())
model.add(Dropout(0.2))
model.add(GRU(128, return_sequences=False, implementation=2))
model.add(Dense(1))

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

# Train the Model
st.sidebar.header("Training Model...")
with st.spinner("Model is training, please wait..."):
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_scheduler, early_stopping])
st.sidebar.success("Model Trained Successfully")

# Chart Pattern Detection
def detect_chart_patterns(data):
    patterns = []
    window = 5

    if not isinstance(data, (pd.Series, np.ndarray, list)) or data.empty:
        return patterns

    data = pd.to_numeric(data, errors='coerce')
    data.dropna(inplace=True)

    for i in range(window, len(data) - window):
        if data.iloc[i] > max(data.iloc[i - window:i]) and data.iloc[i] > max(data.iloc[i + 1:i + window + 1]):
            patterns.append((i, 'Top'))
        elif data.iloc[i] < min(data.iloc[i - window:i]) and data.iloc[i] < min(data.iloc[i + 1:i + window + 1]):
            patterns.append((i, 'Bottom'))
    return patterns

# RSI Calculation
def calculate_rsi(data, period=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Live Prediction with Volatility Calculation
st.subheader("Live Prediction with Buy/Sell Signals")

def calculate_volatility(prices, window=14):
    return np.std(prices[-window:]).item()

while True:
    live_data = yf.download(selected_ticker, period='1d', interval='1m')

    if live_data.empty:
        st.warning("No live data available. Retrying...")
        time.sleep(60)
        continue

    live_price = float(live_data['Close'].iloc[-1])

    latest_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    next_day_prediction = model.predict(latest_sequence)
    next_day_price = scaler.inverse_transform(next_day_prediction)[0, 0]

    st.write(f"Current {selected_ticker_name}: ₹ {live_price:.2f}")
    st.write(f"Predicted Next Price: ₹ {next_day_price:.2f}")

    time.sleep(60)