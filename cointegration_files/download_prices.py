import os
import pandas as pd
import yfinance as yf

# Expanded list of tickers (focus on likely cointegrated pairs and sector pairs)
TICKERS = [
    'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS',
    'XOM', 'CVX', 'COP', 'OXY',
    'KO', 'PEP', 'KDP',
    'V', 'MA', 'AXP',
    'T', 'VZ',
    'UNH', 'PFE',
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD',
    'SPY', 'QQQ', 'DIA', 'IWM', 'XLK', 'XLF', 'XLE', 'XLP'
]

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def download_ticker(ticker):
    df = yf.download(ticker, interval='1d', auto_adjust=True)
    if df is None or df.empty:
        print(f"Warning: No data found for {ticker}, skipping.")
        return None
    df.index = pd.to_datetime(df.index)
    return df[['Close']].rename(columns={'Close': str(ticker)})

# Download all data
dfs = []
for t in TICKERS:
    df = download_ticker(t)
    if df is not None:
        dfs.append(df)

# Align all on the same date index and clean up
if dfs:
    all_prices = pd.concat(dfs, axis=1)
    all_prices = all_prices.sort_index().ffill()
    all_prices.index = all_prices.index.map(lambda x: x.strftime('%Y-%m-%d'))
    all_prices.index.name = 'Date'
    all_prices.reset_index(inplace=True)
    all_prices.to_csv(os.path.join(DATA_DIR, 'all_prices_aligned.csv'), index=False)
    print('Clean, aligned data saved to data/all_prices_aligned.csv!')
else:
    print('No data was downloaded for any ticker.') 