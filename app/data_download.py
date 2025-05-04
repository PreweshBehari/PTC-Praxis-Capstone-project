##########################################################################################
# Imports
##########################################################################################
import os
import json

import pandas as pd
import yfinance as yf
import streamlit as st


##########################################################################################
# Data Download Methods
##########################################################################################
def get_tickers(save_path="data/tickers.json"):
    """
    Retrieves a dictionary of S&P 500 ticker symbols and their company names.

    If a local JSON file exists at `save_path`, it loads and returns the data from the file.
    Otherwise, it scrapes the data from Wikipedia, processes it for Yahoo Finance compatibility,
    saves it to the specified path, and returns the result.

    Args:
        save_path (str): Path to the JSON file where ticker data is saved or loaded from.

    Returns:
        dict: A dictionary where keys are ticker symbols (str) and values are company names (str).
    """
    # If the JSON file already exists, load and return it
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            tickers = json.load(f)
        st.info(f"Loaded {len(tickers)} tickers from {save_path}")
        return tickers

    # Otherwise, fetch the S&P 500 list from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    ticker_codes = table[0]['Symbol'].tolist()      # e.g., ['AAPL', 'MSFT', ...]
    ticker_names = table[0]['Security'].tolist()    # e.g., ['Apple Inc.', 'Microsoft Corp.', ...]

    # Replace dots with dashes in ticker symbols for Yahoo Finance compatibility (e.g., BRK.B -> BRK-B)
    tickers = {code.replace('.', '-'): name for code, name in zip(ticker_codes, ticker_names)}

    # Save the processed tickers to a local JSON file
    with open(save_path, 'w') as f:
        json.dump(tickers, f, indent=4)

    st.info(f"Saved {len(tickers)} tickers to {save_path}")
    return tickers


def download_close_prices(tickers, start_date="2020-01-01", end_date=None):
    """
    Downloads only the 'Close' prices for a list of stock tickers using yfinance.

    This function fetches the adjusted close prices for the given tickers within the specified date range.
    It also provides real-time download progress via Streamlit widgets.

    Args:
        tickers (list of str): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): The start date for historical data in 'YYYY-MM-DD' format.
        end_date (str or None): The end date for historical data. If None, defaults to today's date.

    Returns:
        pd.DataFrame: A DataFrame with dates as index and ticker symbols as columns,
                      containing only the close prices.
    """
    # Initialize Streamlit progress bar and status message
    progress = st.progress(0)
    status_text = st.empty()

    data_dict = {}  # Dictionary to store Close prices for each ticker
    total = len(tickers)  # Total number of tickers

    # Loop through each ticker and download data individually
    for i, ticker in enumerate(tickers):
        status_text.text(f"Downloading {ticker} ({i+1}/{total})...")
        try:
            # Download the historical data
            data = yf.download(ticker, start=start_date, end=end_date,
                               auto_adjust=True, progress=False, threads=True)

            # Extract the 'Close' column and store it in the dictionary
            close_series = data['Close'].copy()
            data_dict[ticker] = close_series
        except Exception as e:
            # Warn if any ticker fails to download
            st.warning(f"Failed to download {ticker}: {e}")

        # Update the Streamlit progress bar
        progress.progress((i + 1) / total)

    # Combine all Close series into one DataFrame with ticker names as column headers
    all_data = pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())

    # Indicate that the download is complete
    st.success("Download complete!")

    return all_data