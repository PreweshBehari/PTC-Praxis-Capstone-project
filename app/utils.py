##########################################################################################
# Imports
##########################################################################################
import hashlib
import requests
import pandas as pd

from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime


##########################################################################################
# Utils
##########################################################################################
def get_sp500_spy_etf(topn=30):
    """
    Retrieves the top N tickers, company names, and weights from the SPY ETF 
    (representing the S&P 500) using data from Slickcharts.

    The SPY ETF closely tracks the S&P 500 index, and this method extracts 
    the top holdings (by weight), which effectively reflect the top stocks 
    by free-float market capitalization.

    Args:
        topn (int, optional): Number of top components to retrieve. Defaults to 30.

    Returns:
        dict: A dictionary where the keys are ticker symbols and the values are company names.
              Example: {'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corp.', ...}
    """
    try:
        # URL of the Slickcharts S&P 500 holdings page
        url = "https://www.slickcharts.com/sp500"

        # Custom headers to mimic a browser and bypass potential scraping blocks
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            )
        }

        # Send HTTP GET request to fetch the web page content
        response = requests.get(url, headers=headers)

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the first HTML table on the page (which contains the holdings)
        table = soup.find("table")

        # Convert the HTML table into a pandas DataFrame
        df = pd.read_html(StringIO(str(table)))[0]

        # Select only the top N rows and the relevant columns
        df_data = df.head(topn)[['Symbol', 'Company', 'Weight']]

        # Convert the weight column from string percentages to float
        df_data['Weight'] = df_data['Weight'].str.replace('%', '').astype(float)

        # Replace dots with dashes in ticker symbols for Yahoo Finance compatibility (e.g., BRK.B -> BRK-B)
        # Replace '.' with '-' in the Symbol column
        df_data['Symbol'] = df_data['Symbol'].str.replace('.', '-', regex=False)

        # st.write(f"Top {topn} S&P 500 constituents by free-float market capitalization:")
        # st.write(df_data)

        # Return a dictionary mapping ticker symbols to company names
        return df_data.set_index("Symbol")["Company"].to_dict()
    
    except Exception as e:
        print(f"Failed to get SP500 SPY ETF from website: {str(e)}")

    # Return an empty dictionary if the fetch fails
    return {}


def create_unique_value(combined_keys):
    """
    Generates a unique hash string from a given input string using MD5.

    This can be used to create unique identifiers from a combination of keys.

    Args:
        combined_keys (str): A string composed of concatenated values to be hashed.

    Returns:
        str: A unique MD5 hash of the input string.
    """
    # Encode the string and generate its MD5 hash
    unique_hash = hashlib.md5(combined_keys.encode()).hexdigest()
    return unique_hash


def human_readable_date(input_date, input_date_format="%Y-%m-%d", output_date_format="%d %B %Y"):
    """
    Converts a date into a human-readable format.

    Useful for displaying dates in reports, charts, or UIs in a more friendly format.

    Args:
        input_date (str or datetime): The date to be formatted. Can be a string or a datetime object.
        input_date_format (str, optional): Format of the input date string. Defaults to "%Y-%m-%d".
        output_date_format (str, optional): Desired output format. Defaults to "%d %B %Y" (e.g., "12 May 2025").

    Returns:
        str: The date formatted in a human-readable way. If parsing fails, returns the original input.
    """
    try:
        # Convert input string to datetime object if needed
        if isinstance(input_date, str):
            date_object = datetime.strptime(input_date, input_date_format)
        else:
            date_object = input_date

        # Format the datetime object to the desired output format
        human_readable = date_object.strftime(output_date_format)
    except Exception as e:
        print(f"\nHuman Readable Date exception: {e}")
        return input_date

    return human_readable

def get_top_n_stocks_by_price_change(price_df, tickers_dict, topn=5):
    """
    Identify and return the top N stocks with the highest absolute price change over the given time period.
    
    Parameters:
    - price_df (pd.DataFrame): DataFrame containing stock prices with dates as the index and stock tickers as columns.
    - tickers_dict (dict): Dictionary mapping stock tickers to their full names.
    - topn (int): Number of top stocks to return based on absolute price change. Default is 5.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the top N stocks with their tickers, names, and price changes.
    """
    # Calculate the price change from the first to the last date for each stock
    abs_change = price_df.iloc[-1] - price_df.iloc[0]
    
    # Create a DataFrame containing tickers, names, and price change values
    change_df = pd.DataFrame({
        'Stock': abs_change.index,
        'Name': [tickers_dict.get(t, t) for t in abs_change.index],
        'Price Change ($)': abs_change.values
    })
    
    # Add a column for absolute price change (used for ranking)
    change_df['Abs Change'] = change_df['Price Change ($)'].abs()
    
    # Sort by absolute price change (descending) and select top N
    top_n_df = change_df.sort_values(by='Abs Change', ascending=False).head(topn).drop(columns='Abs Change')
    
    return top_n_df


def get_top_n_low_volatility_stocks(returns_df, tickers_dict, topn=5):
    """
    Identify and return the top N stocks with the lowest historical volatility (standard deviation of daily returns).

    Parameters:
    - returns_df (pd.DataFrame): DataFrame containing daily returns of stocks, with dates as the index and tickers as columns.
    - tickers_dict (dict): Dictionary mapping stock tickers to full names.
    - topn (int): Number of top stocks with lowest volatility to return. Default is 5.

    Returns:
    - pd.DataFrame: A DataFrame with the tickers, names, and volatility (as percentage) of the top N least volatile stocks.
    """
    # Calculate the standard deviation (volatility) of daily returns for each stock
    volatility = returns_df.std()

    # Create a DataFrame containing tickers, names, and calculated volatility (converted to %)
    volatility_df = pd.DataFrame({
        'Stock': volatility.index,
        'Name': [tickers_dict.get(t, t) for t in volatility.index],
        'Volatility (%)': volatility.values * 100  # Convert from fraction to percentage
    })

    # Sort the DataFrame in ascending order of volatility and select the top N least volatile stocks
    top_n_low_vol_df = volatility_df.sort_values(by='Volatility (%)').head(topn)

    return top_n_low_vol_df
