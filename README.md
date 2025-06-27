# Stock Market Analysis and Prediction Tool

Welcome to the Stock Market Analysis and Prediction Tool, an application designed to help you navigate the dynamic world of the stock market. Whether you're an investor, trader, or simply curious about stock performance, this tool provides the insights and predictions you need to make informed decisions.

## Features

This application is built on real-time data from TwelveData Stocks. It allows the user to perform the following tasks:

- **Predict Stock Prices**: Our predictive model allows you to forecast the price of a stock on a specific date. Say goodbye to uncertainty and make data-driven decisions.

- **Analyze Stock Trends**: Dive deep into historical data and analyze the trends of your favorite stocks. Spot patterns, identify potential opportunities, and gain valuable insights.

- **Compare Stock Performance**: Evaluate and compare the performance of different stocks side by side. Get a comprehensive view of how your investments stack up against each other.

## Local Setup

1. Clone the 'updated' branch of the repository.
    ```bash
    git clone --branch updated https://github.com/AarthiJuryala/Stock-Market-Prediction---Hacktober-fest-23.git
    cd Stock-Market-Prediction---Hacktober-fest-23
    ```
2. Install dependencies. 
    ```bash
    pip install -r requirements.txt
    ```
3. Update the ML Models (Optional)
    ```bash
    python train_prophet_model.py
    ```
4. Run the Streamlit App
    ```bash
    streamlit run StockWebApp.py
    ```

## Demo

Check out the deployed version of our data-driven stock market analysis app on Streamlit: [Stock Market Analysis and Prediction Tool](https://stock-market-prediction-and-analysis.streamlit.app/)
