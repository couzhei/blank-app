import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import talib as ta
from binance.spot import Spot
from datetime import datetime, timedelta
import os

# Load environment variables
BINANCE_API_KEY="wVR8L0aSLXokmsWf0NauyOHI6iqkpxFZjhT2aIrNRNjy8E21RktANP6F2boLjfUB"
BINANCE_SECRET_KEY="9m8YabFlHqnLzUZIkCOb4NIgwJFLBi8mOyxeACcdm0evj7WUgrg3V681UeMQOAK0"

# Function to fetch historical data from Binance
def get_historical_klines(interval: str = "1d", limit: int = 252 * 2, symbol: str = "BTCUSDT") -> pd.DataFrame:
    client = Spot(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        klines,
        columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_asset_volume', 'Number_of_trades',
            'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
        ]
    )
    df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Timestamp'] = df['Timestamp'].astype(float) // 1000
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype("float64")
    return df

# Function to generate signals
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df['sma100'] = ta.SMA(df['Close'], timeperiod=100)
    df['Signal'] = 0

    df['Signal'][100:] = np.where(df['Close'][100:] > df['sma100'][100:], 1, 0)  # Buy signal when close > sma100
    df['Signal'][100:] = np.where(df['Close'][100:] < df['sma100'][100:], -1, df['Signal'][100:])  # Sell signal when close < sma100
    return df

# Function to create annotations for buy/sell signals
def create_annotations(df: pd.DataFrame) -> list:
    annotations = []
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and df['Signal'].iloc[i-1] != 1:
            annotations.append(dict(x=df['Date'].iloc[i], y=df['Close'].iloc[i],
                                    ax=0, ay=-10, xref='x', yref='y',
                                    showarrow=True, arrowhead=2,
                                    text='BUY', font=dict(color='purple')))
        elif df['Signal'].iloc[i] == -1 and df['Signal'].iloc[i-1] != -1:
            annotations.append(dict(x=df['Date'].iloc[i], y=df['Close'].iloc[i],
                                    ax=0, ay=10, xref='x', yref='y',
                                    showarrow=True, arrowhead=2,
                                    text='SELL', font=dict(color='blue')))
    return annotations

# Streamlit app layout
st.set_page_config(layout="wide")
st.title('Real Time Crypto Dashboard')

# Sidebar for user input parameters
st.sidebar.header('Chart Parameters')
symbol = st.sidebar.text_input('Symbol', 'BTCUSDT')
interval = st.sidebar.selectbox('Interval', ['1m', '5m', '15m', '1h', '1d'])
limit = st.sidebar.slider('Limit', 100, 1000, 500)

# Fetch and process data
if st.sidebar.button('Update'):
    # df = get_historical_klines(interval=interval, limit=limit, symbol=symbol)
    df = pd.read_csv("Historical_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = generate_signals(df)
    annotations = create_annotations(df)

    # Create the candlestick chart with SMAs
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'])])

    # Add SMAs to the figure
    fig.add_trace(go.Scatter(x=df['Date'], y=df['sma100'], mode='lines', name='SMA 100', line=dict(color='orange', width=1)))

    # Update layout with annotations and title
    fig.update_layout(
        title=dict(text=f'{symbol} with Buy/Sell Signals'),
        yaxis=dict(title=dict(text='Price')),
        annotations=annotations,
        xaxis_rangeslider_visible=False
    )

    # Show the figure
    st.plotly_chart(fig, use_container_width=True)

    # Display historical data and technical indicators
    st.subheader('Historical Data')
    st.dataframe(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])

    st.subheader('Technical Indicators')
    st.dataframe(df[['Date', 'sma100', 'Signal']])