##########################################################################################
# Imports
##########################################################################################
import os

import streamlit as st
import pandas as pd
import numpy as np

from datetime import date

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
from app.utils import create_unique_value


##########################################################################################
# Data Visualization Methods
##########################################################################################

st.set_page_config(page_title="Efficient Portfolio Builder", layout="wide")

st.title("📈 Efficient Portfolio Builder (Markowitz vs HRP)")

# Main app logic
try:

    start_date = "2020-01-01"
    end_date = date.today().strftime("%Y-%m-%d")

    # Get Tickers
    tickers_dict = get_tickers()

    # Create mapping: name -> ticker
    name_to_ticker = {name: ticker for ticker, name in tickers_dict.items()}

    # List of names to display
    stock_names = list(name_to_ticker.keys())

    # Sidebar
    st.sidebar.header("Portfolio Settings")
    # Streamlit multiselect with display names
    selected_ticker_names = st.sidebar.multiselect(
        "Select stocks", stock_names, default=stock_names[:30]
    )
    method = st.sidebar.radio("Optimization Method", ["HRP", "Markowitz"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime(start_date))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime(end_date))
    max_weight = st.sidebar.slider("Max allocation per stock (%)", 1, 100, 30)

    # Map back to ticker codes
    selected_tickers = [name_to_ticker[name] for name in selected_ticker_names]

    if st.button("Load Stock Prices and Create Portfolios"):
        combined_keys = f"{''.join(selected_ticker_names)}{start_date}-{end_date}"
        stock_price_file_unique_id = create_unique_value(combined_keys)

        stock_price_file = f"data/stock_prices_{stock_price_file_unique_id}.csv"

        # Stock prices CSV already exists, load from cache
        if os.path.exists(stock_price_file):
            close_prices_data = pd.read_csv(stock_price_file)

            st.info(f"Loaded Stock Prices from {stock_price_file}")
        else:
            # Download the stock prices
            st.subheader("Downloading Stock Prices")

            # Download stock prices for the selected stocks
            close_prices_data = download_close_prices(selected_tickers, start_date, end_date)

            # Keep only the second level of column MultiIndex (named 'Ticker')
            close_prices_data.columns = close_prices_data.columns.get_level_values('Ticker')

            # Save to CSV
            close_prices_data.to_csv(stock_price_file)

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
        create_heatmap(close_prices_data)

        st.header("Data Preparation")
        # Checking for missing values and remove the missing values
        st.write("Check for missing values in the dataset")
        any_missing_values = close_prices_data.isnull().values.any()
        st.write(f'Missing Values = {any_missing_values}')

        if any_missing_values:
            # Fill the missing values with the last value available in the dataset.
            # forward-fills missing values (uses the last known value to fill in gaps).
            close_prices_data.ffill(inplace=True)

            # Remove any columns (stocks) that still have missing values after forward fill.
            close_prices_data = drop_columns_with_excessive_missing_data(close_prices_data)

            # Double Checking for missing values
            st.write(f'Missing Values = {close_prices_data.isnull().values.any()}')

        # 4.2 Data Transformation
        st.header("Data Transformation")
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

        # Build Portfolios
        portfolios = build_portfolios(returns)

        # Visualize portfolios
        create_portfolios_piechart(portfolios)

        # 6. Backtesting-Out Of Sample
        st.header("6. Backtesting-Out Of Sample")

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

        # In Sample Results
        st.write("In Sample Results")

        # Calculate in sample results
        insample_results = calculate_sample_metrics(in_sample_result)

        # View the in-sample results
        st.write(insample_results)

        # Out of Sample Results
        st.write("Out of Sample Results")

        # Calculate out of sample results
        outsample_results = calculate_sample_metrics(out_of_sample_result)

        # View the out-sample results
        st.write(outsample_results)



except Exception as e:
    st.error(f"Error: {str(e)}")
