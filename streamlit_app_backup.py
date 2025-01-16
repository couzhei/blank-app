import streamlit as st


import plotly.graph_objects as go
import pandas as pd
import numpy as np
import talib as ta





# Load data
@st.cache_data
def load_dataset():
    df = pd.read_csv("Historical_data.csv", parse_dates=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["BarColor"] = df[["Open", "Close"]].apply(lambda o: "red" if o.Open > o.Close else "green", axis=1)
    df["Date_str"] = df["Date"].astype(str)
    ## Calculate Various Indicators
    df["SMA"] = ta.SMA(df.Close, timeperiod=3)
    df["MA"]  = ta.MA(df.Close, timeperiod=3)
    df["EMA"] = ta.EMA(df.Close, timeperiod=3)
    df["WMA"] = ta.WMA(df.Close, timeperiod=3)
    df["RSI"] = ta.RSI(df.Close, timeperiod=3)
    df["MOM"] = ta.MOM(df.Close, timeperiod=3)
    df["DEMA"] = ta.DEMA(df.Close, timeperiod=3)
    df["TEMA"] = ta.TEMA(df.Close, timeperiod=3)
    # Calculate SMAs
    df['SMA100'] = ta.SMA(df['Close'], timeperiod=100)

    # Generate signals
    df['Signal'] = 0  # Default no signal

    df['Signal'][100:] = np.where(df['Close'][100:] > df['SMA100'][100:], 1, 0)  # Buy signal when close > SMA100
    df['Signal'][100:] = np.where(df['Close'][100:] < df['SMA100'][100:], -1, df['Signal'][100:])  # Sell signal when close < SMA100
        
    return df

df = load_dataset()

df_orig = pd.read_csv("Historical_data.csv")
df = df_orig.copy()

df.Date = pd.to_datetime(df.Date)
df.sort_values(by="Date", ascending=True, inplace=True)

# Calculate SMAs
df['sma100'] = ta.SMA(df['Close'], timeperiod=100)

# Generate signals
df['Signal'] = 0  # Default no signal

df['Signal'][100:] = np.where(df['Close'][100:] > df['sma100'][100:], 1, 0)  # Buy signal when close > sma100
df['Signal'][100:] = np.where(df['Close'][100:] < df['sma100'][100:], -1, df['Signal'][100:])  # Sell signal when close < sma100

# # Generate signals using .loc to avoid ChainedAssignmentError
# df.loc[100:, 'Signal'] = np.where(df['Close'][100:] > df['sma100'][100:], 1, 0)  # Buy signal when close > sma100
# df.loc[100:, 'Signal'] = np.where(df['Close'][100:] < df['sma100'][100:], -1, df.loc[100:, 'Signal'])  # Sell signal when close < sma100


# Create a list for annotations
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

# Create the candlestick chart with SMAs
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                      open=df['Open'], high=df['High'],
                                      low=df['Low'], close=df['Close'])])

# Add SMAs to the figure
fig.add_trace(go.Scatter(x=df['Date'], y=df['sma100'], mode='lines', name='SMA 100', line=dict(color='orange', width=1)))

# Update layout with annotations and title
fig.update_layout(
    title=dict(text='BTC~USDT with Buy/Sell Signals'),
    yaxis=dict(title=dict(text='Price')),
    shapes=[dict(
        x0='2023-09-01', x1='2023-09-01', y0=300, y1=1000, xref='x', yref='paper',
        line_width=2)],
    annotations=annotations,
    xaxis_rangeslider_visible=False
)

# Show the figure
fig.show()


st.plotly_chart(fig, use_container_width=True)