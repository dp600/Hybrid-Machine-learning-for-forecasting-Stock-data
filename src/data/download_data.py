"""
Download OHLCV data with yfinance based on configs/config.yaml
Usage:
  python -m src.data.download_data --config configs/config.yaml
"""

import yfinance as yf
import pandas as pd
# List of tickers
tickers= ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JNJ", "V"]
#Fetch data for each ticker and store in a list
all_data =[]
for ticker in tickers:
  print(f"Fetching data for {ticker}...")
  ticker_data = yf.Ticker(ticker)
  ticker_df=ticker_data.history(period="max")
  ticker_df['Ticker'] = ticker
  all_data.append(ticker_df)

Combined_df = pd.concat(all_data)
print(f'Total number of data points: {len(Combined_df)}')
# Display the first few rows
print(Combined_df.head)
