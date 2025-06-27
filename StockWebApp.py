import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from streamlit_extras.no_default_selectbox import selectbox
from prophet.serialize import model_from_json
import json
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import requests

# ----------------------------
# Company Mapping
# ----------------------------
comp_keys = {
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'BLK': 'BlackRock',
    'COO': 'Cooper Companies',
    'DGX': 'Quest Diagnostics',
    'ETR': 'Entergy Corp',
    'FOX': 'Fox Corp',
    'GS': 'Goldman Sachs',
    'MAC': 'Macerich Co',
    'NFLX': 'Netflix'
}

# ----------------------------
# Helper Functions
# ----------------------------
def load_model(ticker, target):
    model_path = f"models/{ticker}_{target}.json"
    with open(model_path, "r") as f:
        model = model_from_json(json.load(f))
    return model

def get_forecast_for_date(model, future_date_str):
    future_df = model.make_future_dataframe(periods=365)
    forecast = model.predict(future_df)
    row = forecast[forecast['ds'] == future_date_str]
    if not row.empty:
        return row['yhat'].values[0]
    return None

def load_twelvedata(symbol, start_date, end_date):
    API_KEY = st.secrets["twelvedata"]["api_key"]
    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={symbol}&interval=1day&start_date={start_date}&end_date={end_date}&apikey={API_KEY}&outputsize=5000"
    )

    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Error fetching data: {data.get('message', 'Unknown error')}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.rename(columns={
        "datetime": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_values("Date")
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    return df

# ----------------------------
# Sidebar
# ----------------------------
st.markdown("""
    <style>
        .sidebar-content .sidebar-components .stMarkdown {
            font-size: 60px;
            font-weight: bold;
            color:white;
        }
        [data-testid=stSidebar] {
            background-color:#202A44;
        }
        [data-testid="stSidebar"] h2 {
            color: white;
        }
        [data-testid="stSidebar"] [data-testid="stDateInput"] label {
            color: white;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    page = selectbox("Choose Your Task", ["Predict Future Stock Price", "Stock Trends Exploration", "Compare Stocks"])

# ----------------------------
# Main Page Dispatcher
# ----------------------------
if not page:
    st.title("STOCK MARKET ANALYSIS")
    image_path = 'https://img.freepik.com/premium-photo/stock-market-financial-graph-interface-dark-blue-background_269648-475.jpg'
    st.markdown(
        f"""
        <img src="{image_path}" width="800" height="500" style="object-fit: cover">
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Stock Trends Exploration
# ----------------------------
if page == "Stock Trends Exploration":
    st.sidebar.header("Details:")
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    symbol = selectbox('Stock Symbol', list(comp_keys.keys()))

    if start_date and end_date and symbol:
        df = load_twelvedata(symbol, start_date, end_date)
        company_name = comp_keys[symbol]

        st.header(f"{company_name} Forecasted Stock Market")

        for target in ['open', 'close', 'volume']:
            model = load_model(symbol, target)
            p = (end_date - start_date).days
            future = model.make_future_dataframe(periods=p)
            forecast = model.predict(future)
            fig = go.Figure()
            # Filter forecast to selected date range
            mask = (forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))
            filtered_forecast = forecast[mask]
            fig.add_trace(go.Scatter(x=filtered_forecast['ds'], y=filtered_forecast['yhat'], name=f'Forecast {target.title()}'))
            fig.update_layout(title=f'{company_name} {target.title()} Forecast', xaxis_title='Date', yaxis_title=target.title())
            st.plotly_chart(fig)

        st.header(f"{company_name} History")
        for column in ['Open', 'Close', 'Volume']:
            st.subheader(f"{company_name} {column} Price")
            st.line_chart(df[column])

# ----------------------------
# Predict Future Stock Price
# ----------------------------
if page == "Predict Future Stock Price":
    st.title("Predict Future Stock Price")
    future_date = st.date_input('Select a future date', min_value=datetime.now() + timedelta(days=1), value=datetime.now() + timedelta(days=1))
    symbol = selectbox('Select a stock', list(comp_keys.keys()))
    future_date_str = future_date.strftime('%Y-%m-%d')

    if st.button('Predict'):
        try:
            model_open = load_model(symbol, 'open')
            model_close = load_model(symbol, 'close')
            model_volume = load_model(symbol, 'volume')

            pred_open = get_forecast_for_date(model_open, future_date_str)
            pred_close = get_forecast_for_date(model_close, future_date_str)
            pred_volume = get_forecast_for_date(model_volume, future_date_str)

            if pred_open is not None and pred_close is not None and pred_volume is not None:
                st.success(f"**{comp_keys[symbol]} Prediction for {future_date_str}:**")
                st.write(f"Open Price: **${pred_open:.2f}**")
                st.write(f"Close Price: **${pred_close:.2f}**")
                st.write(f"Volume: **{pred_volume:,.0f}**")
            else:
                st.warning("Prediction not available for selected date.")

        except FileNotFoundError:
            st.error(f"Model files for {symbol} not found.")

# ----------------------------
# Compare Stocks
# ----------------------------
if page == "Compare Stocks":
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date",max_value=datetime.now())
    st.title('Compare Stock Prices')
    stock1 = selectbox('Select the first stock', list(comp_keys.keys()), key='s1')
    stock2 = selectbox('Select the second stock', list(comp_keys.keys()), key='s2')

    if stock1 and stock2:
        df1 = load_twelvedata(stock1, start_date,end_date)
        df2 = load_twelvedata(stock2, start_date,end_date)

        for column, label in zip(['Open', 'Close', 'Volume'], ['Opening Price', 'Closing Price', 'Volume']):
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.set_facecolor('black')
            ax.set_facecolor('black')
            ax.plot(df1["Date"], df1[column], label=f'{comp_keys[stock1]} {label}')
            ax.plot(df2["Date"], df2[column], label=f'{comp_keys[stock2]} {label}')
            ax.set_xlabel('Date', color='white')
            ax.set_ylabel(label, color='white')
            ax.legend()
            ax.tick_params(axis='both', colors='white')
            ax.grid(True)
            st.pyplot(fig)
