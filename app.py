##########################################################################################
# Imports
##########################################################################################
import os

import streamlit as st
import pandas as pd
import numpy as np

from datetime import date
from dateutil.relativedelta import relativedelta

from collections import OrderedDict

from scipy.cluster.hierarchy import linkage

from app.optimizer import correlDist, build_portfolios, calculate_sample_metrics
from app.data_visualization import (
    create_heatmap,
    create_dendrogram,
    create_portfolios_piechart,
    create_in_and_out_sample_plots,
)
from app.data_preparation import drop_columns_with_excessive_missing_data
from app.data_download import get_tickers, download_close_prices
from app.data_transformation import split_dataset
from app.portfolio_insights import display_insights, recommend_stocks
from app.utils import create_unique_value, human_readable_date, get_sp500_spy_etf


##########################################################################################
# Data Visualization Methods
##########################################################################################

# Settings
st.set_page_config(page_title="Efficient Portfolio Builder", layout="wide")

st.title("📈 Efficient Portfolio Builder (Markowitz vs HRP)")

# Initialize in session_state if not already there
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'stock_price_file' not in st.session_state:
    st.session_state.stock_price_file = None
if 'user_uploaded_file' not in st.session_state:
    st.session_state.user_uploaded_file = False
if 'create_portfolio_clicked' not in st.session_state:
    st.session_state.create_portfolio_clicked = False

# Get today's date
date_today_obj = date.today()

# Calculate start date as 5 years before date_today
start_date_obj = date_today_obj - relativedelta(years=5)

# Format as string
start_date = start_date_obj.strftime("%Y-%m-%d")
end_date = date_today_obj.strftime("%Y-%m-%d")

# Fetch the top N tickers, company names, and weights from the SPY ETF via Slickcharts.
# These represent the largest S&P 500 constituents by free-float market capitalization.
tickers_with_highest_weights = get_sp500_spy_etf()

# Retrieve a broader set of S&P 500 tickers and company names from the Wikipedia page.
tickers_dict = get_tickers()

# Merge both dictionaries while preserving order:
# - Ensure tickers from Slickcharts (top-weighted) appear first.
# - Avoid duplicates by excluding any keys from the Wikipedia data already present in the top-weighted list.
if tickers_with_highest_weights:
    merged_tickers_dict = OrderedDict()
    merged_tickers_dict.update(tickers_with_highest_weights)
    merged_tickers_dict.update({k: v for k, v in tickers_dict.items() if k not in merged_tickers_dict})
else:
    # Fall back to using only Wikipedia data if the Slickcharts request failed or returned empty.
    merged_tickers_dict = tickers_dict

# Create mapping: name -> ticker
name_to_ticker = {name: ticker for ticker, name in merged_tickers_dict.items()}

# List of names to display
stock_names = list(name_to_ticker.keys())

# Sidebar
st.sidebar.header("Portfolio Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start_date))
end_date = st.sidebar.date_input("End Date", pd.to_datetime(end_date))
# Add a slider to configure the threshold
threshold = st.sidebar.slider(
    "Minimum allocation threshold (%)",
    min_value=0.0,
    max_value=30.0,
    value=0.5,
    step=0.1,
    help="Only stocks with an allocation above this percentage will be recommended."
)
# Streamlit multiselect with display names
selected_ticker_names = st.sidebar.multiselect(
    "Select stocks", stock_names, default=stock_names[:30]
)

# Minimal two stocks should be selected
if len(selected_ticker_names) < 2:
    st.error("Select atleast two stocks from the selectbox")
    st.stop()

# Map back to ticker codes
selected_tickers = [name_to_ticker[name] for name in selected_ticker_names]

st.header(f"1. Load the Data")

# 3 columns: left, middle, right
col1, col_mid, col2 = st.columns([1, 1, 1], border=True)

# Variables to check which button is clicked
with col1:
    if st.button("Upload CSV"):
        st.session_state.mode = 'upload'

with col_mid:
    st.markdown("<div style='text-align: center; padding-top: 0.5em; font-weight: bold;'>------- OR -------</div>", unsafe_allow_html=True)

with col2:
    if st.button("Download Stock Price"):
        st.session_state.mode = 'download'

# Upload CSV logic
if st.session_state.mode == 'upload':
    st.session_state.create_portfolio_clicked = False # Clear Clicked state

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.user_uploaded_file = True
        st.session_state.stock_price_file = uploaded_file
        st.session_state.create_portfolio_clicked = True

# Download Stock Price logic
elif st.session_state.mode == 'download':
    st.session_state.create_portfolio_clicked = False # Clear Clicked state
    st.session_state.user_uploaded_file = False

    combined_keys = f"{''.join(selected_ticker_names)}{start_date}-{end_date}"
    stock_price_file_unique_id = create_unique_value(combined_keys)

    stock_price_file = f"data/stock_prices_{stock_price_file_unique_id}.csv"

    # Download the stock prices
    st.subheader(f"Downloading Stock Prices ({human_readable_date(start_date)} - {human_readable_date(end_date)})")

    # Download stock prices for the selected stocks
    close_prices_data = download_close_prices(selected_tickers, start_date, end_date, merged_tickers_dict)

    # Keep only the second level of column MultiIndex (named 'Ticker')
    close_prices_data.columns = close_prices_data.columns.get_level_values('Ticker')

    # Save to CSV
    close_prices_data.to_csv(stock_price_file)

    st.session_state.stock_price_file = stock_price_file
    st.session_state.create_portfolio_clicked = True


# Main app logic
try:
    if st.session_state.create_portfolio_clicked:
        user_uploaded_file = st.session_state.user_uploaded_file
        stock_price_file = st.session_state.stock_price_file

        # Use uploaded file
        if user_uploaded_file:
            st.info(f"Loaded Stock Prices from uploaded file: {stock_price_file.name}")

            close_prices_data = pd.read_csv(stock_price_file)
        # Stock prices CSV already exists, load from cache
        elif os.path.exists(stock_price_file):
            st.info(f"Loaded Stock Prices from {stock_price_file} ({human_readable_date(start_date)} - {human_readable_date(end_date)})")

            close_prices_data = pd.read_csv(stock_price_file)
        else:
            st.error("No Stock Price data found.")

        # EDA
        st.header("2. Exploratory Data Analysis")

        # If 'Date' is a column, move it to the index
        if 'Date' in close_prices_data.columns:
            close_prices_data = close_prices_data.set_index('Date')
            
        # Data Overview
        # Display the first 5 rows
        st.write("Displaying few rows:")
        st.write(close_prices_data.head())
        st.write(f"The table has a total of {close_prices_data.shape[0]} rows and {close_prices_data.shape[1]} columns")
        # st.write("Data Info:")
        # buffer = io.StringIO()
        # close_prices_data.info(buf=buffer)
        # st.text(buffer.getvalue())
            
        # Create Heatmap
        create_heatmap(close_prices_data, topn=30)

        st.header("3. Data Preparation")
        # Checking for missing values and remove the missing values
        st.write("Check for missing values in the dataset")
        any_missing_values = close_prices_data.isnull().values.any()

        if any_missing_values:
            st.warning("There are missing values.")

            # Fill the missing values with the last value available in the dataset.
            # forward-fills missing values (uses the last known value to fill in gaps).
            close_prices_data.ffill(inplace=True)

            # Remove any columns (stocks) that still have missing values after forward fill.
            close_prices_data = drop_columns_with_excessive_missing_data(close_prices_data)

            # Double Checking for missing values
            if close_prices_data.isnull().values.any():
                st.error("There are still missing values.")
        else:
            st.success("There are no missing values.")

        # 4 Data Transformation
        st.header("4. Data Transformation")
        st.info("""
            For the purpose of clustering, we will be using annual returns.
            Additionally, we will train the data followed by testing.
            Let us prepare the dataset for training and testing, by separating 20% of the dataset
            for testing followed by generating the return series.
        """)
        
        # Call the method to split dataset
        X, train_len = split_dataset(close_prices_data)

        # Splits the dataset into training and testing sets using time-ordered rows (no shuffling).
        X_train = X.head(train_len) # First N rows
        X_test = X.tail(len(X) - train_len) # Remaining rows

        # Calculate percentage return for training set
        returns = X_train.pct_change().dropna()

        st.write("Displaying few Train returns:")
        st.write(returns.head())

        # Calculate percentage return for test set
        returns_test = X_test.pct_change().dropna()

        st.write("Displaying few Test returns:")
        st.write(returns_test.head())

        # 5. Evaluate Algorithms and Models
        st.header("5. Evaluate Algorithms and Models")

        # 5.1. Building Hierarchy Graph/ Dendogram
        st.subheader("5.1. Building Hierarchy Graph/ Dendogram")
        st.info("""
            The first step is to look for clusters of correlations using the agglomerate hierarchical clustering technique. The hierarchy class
            has a dendrogram method which takes the value returned by the linkage method of
            the same class.

            Linkage does the actual clustering in one line of code, and returns a list of the clusters joined in the format: Z = [stock_1, stock_2, distance, sample_count]
        """)

        # Compute how close the stocks are, clustering the most similar ones first,
        # and creating a structure (linkage) to eventually allocate portfolio weights smarter.

        # Calculate the distance matrix from the correlation matrix
        dist = correlDist(returns.corr())

        # Perform hierarchical clustering (linkage) using the Ward linkage method on the distance matrix
        # Ward’s method tries to minimize the variance within clusters at each step.
        link = linkage(dist, 'ward')

        # Show the first linkage step
        st.write("Show the first linkage step")
        st.write(link[0])

        st.info("""
            Computation of linkages is followed by the visualization of the clusters through a dendrogram, which displays a cluster tree.
            The leaves are the individual stocks and the root is the final single cluster. The “distance” between each cluster is shown on the y-axis, and thus the longer the branches
            are, the less correlated two clusters are.
        """)

        # Create Dendrogram
        create_dendrogram(link, X)

        # 5.2 Steps for Hierarchial Risk Parity
        st.subheader("5.2 Steps for Hierarchial Risk Parity")

        #### Quasi-diagonalization and getting the weights for Hierarchial Risk Parity
        st.write("Quasi-diagonalization and getting the weights for Hierarchial Risk Parity")

        st.info("""
            A 'quasi-diagonalization' is a process usually known as matrix seriation and which can be performed using hierarchical clustering.
            This process reorganize the covariance matrix so similar investments will be placed together.
            This matrix diagonalization allow us to distribute weights optimally following an inverse-variance allocation.
        """)

        #### Comparison against other asset allocation methods:
        st.write("Comparison against other asset allocation methods:")

        st.markdown("""
            The main premise of this case study
            was to develop an alternative to Markowitz’s Minimum-Variance Portfolio based
            asset allocation. So, in this step, we define a functions to compare the performance of the following asset allocation methods.

                1. MVP - Markowitz’s Minimum-Variance Portfolio
                2. HRP - Hierarchial Risk Parity
        """)

        st.subheader("5.3 Getting the portfolio weights for all types of asset allocation")

        # Build Portfolios
        portfolios = build_portfolios(returns)

        # Visualize portfolios
        create_portfolios_piechart(portfolios)

        # 6. Backtesting-Out Of Sample
        st.header("6. Backtesting-Out Of Sample")

        st.info("Take the portfolio weights, multiply them with the stock returns, and get the daily portfolio returns for both training and testing periods.")

        # Compute portfolio returns in-sample and out-of-sample
        st.subheader("6.1 In Sample and Out of Sample Results")

        # In-sample: multiply asset returns by portfolio weights
        in_sample_result = pd.DataFrame(
            np.dot(returns, np.array(portfolios)), # matrix multiplication
            columns=['MVP', 'HRP'], # Name the portfolios
            index=returns.index # Keep the same dates
        )

        # Out-of-sample: multiply test asset returns by portfolio weights
        out_of_sample_result = pd.DataFrame(
            np.dot(returns_test, np.array(portfolios)),
            columns=['MVP', 'HRP'],
            index=returns_test.index
        )

        # Create In and Out Sample Plots
        create_in_and_out_sample_plots(in_sample_result, out_of_sample_result)

        ### In Sample and Out of Sample Results
        st.subheader("6.2 In Sample and Out of Sample Results")

        # # In Sample Results
        # st.write("In Sample Results")

        # Calculate in sample results
        insample_results = calculate_sample_metrics(in_sample_result)

        # # View the in-sample results
        # st.write(insample_results)

        # # Out of Sample Results
        # st.write("Out of Sample Results")

        # Calculate out of sample results
        outsample_results = calculate_sample_metrics(out_of_sample_result)

        # # View the out-sample results
        # st.write(outsample_results)

        # Display Insights
        display_insights(insample_results, outsample_results)

        # 7. Recommended Stocks
        st.header("7. Recommended Stocks")

        # Using HRP Strategy
        st.markdown(f"#### Recommended Stocks – HRP Strategy")
        recommend_stocks(portfolios, strategy_name="HRP", threshold=threshold/100, tickers=merged_tickers_dict)

        # Using MVP Strategy
        st.markdown(f"#### Recommended Stocks – MVP Strategy")
        recommend_stocks(portfolios, strategy_name="MVP", threshold=threshold/100, tickers=merged_tickers_dict)

except Exception as e:
    st.error(f"Error: {str(e)}")
