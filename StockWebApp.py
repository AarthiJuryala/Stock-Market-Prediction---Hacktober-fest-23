import streamlit as st
import pandas as pd
from datetime import datetime,timedelta 
from all_models import *
import yfinance as yf
from streamlit_extras.no_default_selectbox import selectbox


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
st.markdown("""
<style>
            .sidebar-content .sidebar-components .stMarkdown {
            font-size: 60px;
            font-weight: bold;
            }
    [data-testid=stSidebar] {
        background-color:#202A44;
    }
    <style>
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
  # Create the selectbox
    page = selectbox("Choose Your Task", ["Predict Future Stock Price", "Stock Trends Exploration", "Compare Stocks"])

if not page:
    st.title("STOCK MARKET ANALYSIS")
    image_path = 'https://img.freepik.com/premium-photo/stock-market-financial-graph-interface-dark-blue-background_269648-475.jpg'  # Replace with your image URL
    full_screen_width = 800
    full_screen_height = 500
    st.markdown(
        f"""
        <style>
        .full-screen-image {{
            width: {full_screen_width}px;
            height: {full_screen_height}px;
            object-fit: cover; /* Preserve aspect ratio while filling the screen */
        }}
        </style>
        <img class="full-screen-image" src="{image_path}" alt="Full Screen Image">
        """,
        unsafe_allow_html=True)
def get_company_name(symbol):
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
        return comp_keys[symbol]

if page =="Stock Trends Exploration":
    st.sidebar.header("Details:")
    def get_input():
        start_date = st.sidebar.date_input("Start Date",datetime.strptime('2019-01-01', '%Y-%m-%d').date())
        end_date = st.sidebar.date_input("End Date",datetime.strptime('2023-10-08', '%Y-%m-%d').date())
        stock_symbol = selectbox('Stock Symbol',('AAPL', 'AMZN', 'BLK', 'COO', 'DGX', 'ETR', 'FOX', 'GS', 'MAC', 'NFLX'))
        return start_date, end_date, stock_symbol

    def get_data(symbol, start_date, end_date):
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        df = yf.download(symbol, start=start_date, end=end_date)
        print(df)
        return df

    start, end, symbol = get_input()

    if(start and end and symbol):
        df = get_data(symbol, start, end)
        company_name = get_company_name(symbol.upper())
        st.header(company_name + " Forecasted Stock Market\n")
        m,forecast1=forecast_open(symbol,df)
        st.plotly_chart(m)
        m,forecast2=forecast_data(symbol,df)
        st.plotly_chart(m)
        m,forecast3=forecast_volume(symbol,df)
        st.plotly_chart(m)
        st.header(company_name + " History\n")
        st.markdown(company_name + " Open Price\n")
        st.line_chart(df['Open'])
        
        st.markdown(company_name + " Close Price\n")
        st.line_chart(df['Close'])

        st.markdown(company_name + " Volume\n")
        st.line_chart(df['Volume'])
        
if page=="Predict Future Stock Price":
        st.title("Predict Future Stock Price")
        future_date = st.date_input('Select a future date', min_value=datetime.now() + timedelta(days=1), value=datetime.now() + timedelta(days=1))
        symbol = selectbox('Select a stock', list(comp_keys.keys()))
        start_date='2019-01-01'
        end_date='2023-10-08'
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if symbol:
            df = yf.download(symbol, start=start_date, end=end_date)
            m,forecast1=forecast_open(symbol,df)
            m,forecast2=forecast_data(symbol,df)
            m,forecast3=forecast_volume(symbol,df)
        if st.button('Predict'):
            future_date_str = future_date.strftime('%Y-%m-%d')
            future_forecast1 = forecast1[forecast1['ds'] == future_date_str]['yhat'].values[0]
            st.write(f'**Predicted {comp_keys[symbol]} Stock Opening Price on {future_date_str}: ${future_forecast1:.2f}**')
            future_forecast2 = forecast2[forecast2['ds'] == future_date_str]['yhat'].values[0]
            st.write(f'**Predicted {comp_keys[symbol]} Stock Closing Price on {future_date_str}: ${future_forecast2:.2f}**')
            future_forecast3 = forecast3[forecast3['ds'] == future_date_str]['yhat'].values[0]
            st.write(f'**Predicted {comp_keys[symbol]} Stock Volume on {future_date_str}: {future_forecast3:.2f}**')

if page=="Compare Stocks":
        st.title('Compare Stock Prices')
        # Select two stocks for comparison
        selected_stock_1 = selectbox('Select the first stock', list(comp_keys.keys()))
        selected_stock_2 = selectbox('Select the second stock', list(comp_keys.keys()))
        if selected_stock_2 and selected_stock_1:
        # Load historical stock data for the selected stocks
            stock_data_1 = yf.download(selected_stock_1, period='1y')
            stock_data_2 = yf.download(selected_stock_2, period='1y')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.set_facecolor('black') 
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')  
            ax.spines['left'].set_color('white') 
            ax.plot(stock_data_1.index, stock_data_1['Open'], label=f'{comp_keys[selected_stock_1]} Opening Price')
            ax.plot(stock_data_2.index, stock_data_2['Open'], label=f'{comp_keys[selected_stock_2]} Opening Price')
            ax.set_xlabel('Date',color='white')
            ax.set_ylabel('Stock Opening Price (USD)',color='white')
            ax.legend()
            ax.tick_params(axis='both', colors='white')
            ax.grid(True)

            # Display the comparison plot
            st.pyplot(fig)

            # Create a Matplotlib figure for comparing stock prices
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.set_facecolor('black') 
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')  
            ax.spines['left'].set_color('white')
            ax.plot(stock_data_1.index, stock_data_1['Close'], label=f'{comp_keys[selected_stock_1]} Closing Price')
            ax.plot(stock_data_2.index, stock_data_2['Close'], label=f'{comp_keys[selected_stock_2]} Closing Price')
            ax.set_xlabel('Date',color='white')
            ax.set_ylabel('Stock Closing Price (USD)',color='white')
            ax.legend()
            ax.tick_params(axis='both', colors='white')
            ax.grid(True)

            # Display the comparison plot
            st.pyplot(fig)

            # Create a Matplotlib figure for comparing stock prices
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.set_facecolor('black') 
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')  
            ax.spines['left'].set_color('white')
            ax.plot(stock_data_1.index, stock_data_1['Volume'], label=f'{comp_keys[selected_stock_1]} Volume')
            ax.plot(stock_data_2.index, stock_data_2['Volume'], label=f'{comp_keys[selected_stock_2]} Volume ')
            ax.set_xlabel('Date',color='white')
            ax.set_ylabel('Number of stocks',color='white')
            ax.legend()
            ax.tick_params(axis='both', colors='white')
            ax.grid(True)

            # Display the comparison plot
            st.pyplot(fig)
            
