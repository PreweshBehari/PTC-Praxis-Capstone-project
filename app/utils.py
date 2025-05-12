##########################################################################################
# Imports
##########################################################################################
import hashlib
import requests
import pandas as pd
import streamlit as st

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