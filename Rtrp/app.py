import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.set_page_config(page_title="Stock Trend Predictor", layout="wide")

# Load trained trend model
model = load_model("stock_trend_model.keras")

st.title("📈 AI Stock Trend Prediction (3-Day Forecast)")

stock = st.text_input("Enter Stock Symbol", "GOOGL")

start = '2010-01-01'
end = '2023-12-31'

data = yf.download(stock, start=start, end=end)

if data.empty:
    st.error("Invalid stock symbol.")
    st.stop()

# --- Feature Engineering (MUST MATCH TRAINING) ---


def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(10).std()
data['Momentum_5'] = data['Close'] - data['Close'].shift(5)
data['Momentum_10'] = data['Close'] - data['Close'].shift(10)
data['MA_20'] = data['Close'].rolling(20).mean()
data['MA_50'] = data['Close'].rolling(50).mean()
data['RSI'] = compute_RSI(data['Close'])
data['Volume_Change'] = data['Volume'].pct_change()

data.dropna(inplace=True)

features = [
    'Close',
    'Return',
    'Volatility',
    'Momentum_5',
    'Momentum_10',
    'MA_20',
    'MA_50',
    'RSI',
    'Volume_Change'
]

sequence_length = 30

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Prepare last 30 days for prediction
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

# Make prediction
probability = model.predict(last_sequence)[0][0]

st.subheader("Prediction for Next 3 Days")

if probability > 0.5:
    st.success(f"📈 UP Trend Predicted")
else:
    st.error(f"📉 DOWN Trend Predicted")

st.write(f"Confidence: {round(float(probability)*100, 2)} %")

# Show raw data
st.subheader("Recent Stock Data")
st.dataframe(data.tail())
