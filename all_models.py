import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pickle
from prophet.serialize import model_to_json, model_from_json
import streamlit as st

comp_keys = {
    'AAPL':'Apple',
    'AMZN':'Amazon',
    'BLK':'BlackRock',
    'COO':'Cooper Companies',
    'DGX':'Quest Diagnostics',
    'ETR':'Entergy CorpEntergy Corp',
    'FOX':'Fox Corp',
    'GS':'Goldman Sachs',
    'MAC':'Macerich Co',
    'NFLX':'Netflix'
}
# Load historical stock data using yfinance
def forecast_data(t,df):
    ticker = t      
    stock_data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    print(stock_data)
    # Initialize and fit the Prophet model
    prophet_model = Prophet()
    #prophet_model.add_regressor('Adj Close')
    #prophet_model.add_regressor('Volume', prior_scale=0.5, mode='multiplicative')

    prophet_model.fit(stock_data)

    model_json = model_to_json(prophet_model)

    # Save the model JSON to a file
    with open('prophet_model.json', 'wb') as f:
        pickle.dump(model_json, f)

    # Later, when you want to use the model
    with open('prophet_model.json', 'rb') as f:
        model_json = pickle.load(f)

    # Re-create the Prophet model from the serialized JSON
    m = model_from_json(model_json)
    # Create a dataframe for future dates
    future = prophet_model.make_future_dataframe(periods=365)
    #future['Adj Close'] = [150.0, 151.0, 152.0, 153.0, 154.0,155.0] 
    #future['Volume'] = [1000000, 1100000, 1050000, 950000, 980000,990000]
    
    
    forecast = m.predict(future)

   # Create a Matplotlib figure and axis for the stock data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['ds'], stock_data['y'], label='Stock Price (USD)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')
    ax.grid(True)

    # Create a Plotly trace graph
    trace = go.Scatter(
        x=stock_data['ds'],
        y=stock_data['y'],
        mode='lines',
        name='Stock Price (USD)',
    )

    # Append the Plotly trace to the Matplotlib plot
    layout = go.Layout(title=f'{comp_keys[ticker]} Closing Stock Price', xaxis=dict(title='Date'), yaxis=dict(title='Stock Price (USD)'))
    fig = go.Figure(data=[trace], layout=layout)

    return fig,forecast
    
def forecast_open(t,df):
    ticker = t      
    stock_data = df.reset_index()[['Date', 'Open']].rename(columns={'Date': 'ds', 'Open': 'y'})

    print(stock_data)
    # Initialize and fit the Prophet model
    prophet_model = Prophet()
    #prophet_model.add_regressor('Adj Close')
    #prophet_model.add_regressor('Volume', prior_scale=0.5, mode='multiplicative')

    prophet_model.fit(stock_data)

    model_json = model_to_json(prophet_model)

    # Save the model JSON to a file
    with open('prophet_model.json', 'wb') as f:
        pickle.dump(model_json, f)

    # Later, when you want to use the model
    with open('prophet_model.json', 'rb') as f:
        model_json = pickle.load(f)

    # Re-create the Prophet model from the serialized JSON
    m = model_from_json(model_json)
    # Create a dataframe for future dates
    future = prophet_model.make_future_dataframe(periods=365)
    #future['Adj Close'] = [150.0, 151.0, 152.0, 153.0, 154.0,155.0] 
    #future['Volume'] = [1000000, 1100000, 1050000, 950000, 980000,990000]
    
    
    forecast = m.predict(future)

   # Create a Matplotlib figure and axis for the stock data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['ds'], stock_data['y'], label='Stock Price (USD)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')
    ax.grid(True)

    # Create a Plotly trace graph
    trace = go.Scatter(
        x=stock_data['ds'],
        y=stock_data['y'],
        mode='lines',
        name='Stock Price (USD)',
    )

    # Append the Plotly trace to the Matplotlib plot
    layout = go.Layout(title=f'{comp_keys[ticker]} Opening Stock Price', xaxis=dict(title='Date'), yaxis=dict(title='Stock Price (USD)'))
    fig = go.Figure(data=[trace], layout=layout)

    return fig,forecast

def forecast_volume(t,df):
    ticker = t      
    stock_data = df.reset_index()[['Date', 'Volume']].rename(columns={'Date': 'ds', 'Volume': 'y'})

    print(stock_data)
    # Initialize and fit the Prophet model
    prophet_model = Prophet()
    #prophet_model.add_regressor('Adj Close')
    #prophet_model.add_regressor('Volume', prior_scale=0.5, mode='multiplicative')

    prophet_model.fit(stock_data)

    model_json = model_to_json(prophet_model)

    # Save the model JSON to a file
    with open('prophet_model.json', 'wb') as f:
        pickle.dump(model_json, f)

    # Later, when you want to use the model
    with open('prophet_model.json', 'rb') as f:
        model_json = pickle.load(f)

    # Re-create the Prophet model from the serialized JSON
    m = model_from_json(model_json)
    # Create a dataframe for future dates
    future = prophet_model.make_future_dataframe(periods=365)
    #future['Adj Close'] = [150.0, 151.0, 152.0, 153.0, 154.0,155.0] 
    #future['Volume'] = [1000000, 1100000, 1050000, 950000, 980000,990000]

    forecast = m.predict(future)

   # Create a Matplotlib figure and axis for the stock data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['ds'], stock_data['y'], label='Stock Price (USD)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price (USD)')
    ax.grid(True)

    # Create a Plotly trace graph
    trace = go.Scatter(
        x=stock_data['ds'],
        y=stock_data['y'],
        mode='lines',
        name='Stock Price (USD)',
    )

    # Append the Plotly trace to the Matplotlib plot
    layout = go.Layout(title=f'{comp_keys[ticker]} Stock Volume', xaxis=dict(title='Date'), yaxis=dict(title='Volume(Units)'))
    fig = go.Figure(data=[trace], layout=layout)

    return fig,forecast
