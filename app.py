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
from scipy.spatial.distance import squareform

# import app.mvp_optimizer as mvp_opt
from app.optimizer import (
    correlDist,
    build_portfolios,
    calculate_sample_metrics,
    simulate_random_portfolios,
    simulate_random_portfolios2,
    display_ef_with_selected,
    efficient_frontier,
)
from app.data_visualization import (
    create_heatmap,
    create_cov_heatmap,
    create_dendrogram,
    create_portfolios_piechart,
    create_in_and_out_sample_plots,
    plot_efficient_frontier,
    plot_efficient_frontier2,
    create_daily_returns_plot,
    create_stock_price_evolved_plot,
    create_portfolio_optimization_plot,
)
from app.data_preparation import drop_columns_with_excessive_missing_data
from app.data_download import get_tickers, download_close_prices
from app.data_transformation import split_dataset
from app.portfolio_insights import display_insights, recommend_stocks
from app.efficient_frontier import get_efficient_portfolio
from app.utils import *


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

# Add a slider for the risk-free rate
risk_free_rate = st.sidebar.slider(
    label="Select Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=1.50, # Default value
    step=0.01,
    help="The risk-free rate is the return of a theoretically riskless investment, often based on government bonds like U.S. Treasury bills."
)

# Add a slider to configure the threshold
threshold = st.sidebar.slider(
    "Select Minimum allocation threshold (%)",
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

    # selected_tickers = list(tickers_dict.keys()) # All tickers
    combined_keys = f"{''.join(selected_tickers)}{start_date}-{end_date}"
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

        # Correct the Column Name
        # The ticker for Berkshire Hathaway Class B is usually returned as 'BRK-B',
        # but sometimes special characters like '-' are replaced with '.'.
        close_prices_data.columns = [col.replace('.', '-') for col in close_prices_data.columns]

        # If 'Date' is a column, move it to the index
        if 'Date' in close_prices_data.columns:
            close_prices_data = close_prices_data.set_index('Date')

        # Select the data of the selected tickers only
        close_prices_data = close_prices_data[selected_tickers]
   
        # EDA
        st.header("2. Exploratory Data Analysis")

        # Data Overview
        # Display the first 5 rows
        st.write("Displaying few rows:")
        st.write(close_prices_data.head())
        st.write(f"The table has a total of {close_prices_data.shape[0]} rows and {close_prices_data.shape[1]} columns")
        # st.write("Data Info:")
        # buffer = io.StringIO()
        # close_prices_data.info(buf=buffer)
        # st.text(buffer.getvalue())

        # Calculate returns
        returns = close_prices_data.pct_change(fill_method=None).dropna()
            
        # Create Heatmap
        st.subheader("Correlation Heatmap")
        create_heatmap(close_prices_data, topn=30)

        # Covariance Matrix
        st.subheader("Covariance Matrix of Returns")
        cov_matrix = returns.cov()
        st.dataframe(cov_matrix.style.format("{:.6f}"))

        # Create Covariance Matrix Heatmap
        st.subheader("Covariance Matrix Heatmap")
        create_cov_heatmap(cov_matrix)

        # Show how Stock price evolved for each stock
        st.markdown("#### Stock price evolving over time")
        st.write("Let's first look at how the price of each stock has evolved within given time frame.")
        create_stock_price_evolved_plot(close_prices_data, tickers_dict)

        # Display the top 5 stocks with the highest absolute price change between first and last date.
        top_n_df = get_top_n_stocks_by_price_change(close_prices_data, tickers_dict)
        st.markdown("##### Top 5 Stocks by Price Change")
        st.dataframe(top_n_df, use_container_width=True, hide_index=True)

        # Calculate percentage return for dataset and plot daily returns
        st.markdown("#### Daily Returns")
        st.write("Another way to plot this is plotting daily returns (percent change compared to the day before). " \
        "By plotting daily returns instead of actual prices, we can see the stocks' volatility.")
        create_daily_returns_plot(returns, tickers_dict)

        # Display the top 5 stocks with the lowest volatility (least daily return fluctuation)
        top_n_df_returns = get_top_n_low_volatility_stocks(returns, tickers_dict)
        st.markdown("##### Top 5 Stocks with the lowest volatility (least daily return fluctuation)")
        st.dataframe(top_n_df_returns, use_container_width=True, hide_index=True)

        st.markdown("#### Random Portfolios Generation")

        st.info("Here we generate 25,000 random portfolios by assigning random weights to each asset (ensuring the total allocation = 100%), " \
        f" using a Risk-Free Rate of **{risk_free_rate:.2f} %**")

        num_portfolios = 25000
        risk_free_rate /= 100 # Divide by 100
        # results, sdp, rp, sdp_min, rp_min = simulate_random_portfolios(returns, num_portfolios, risk_free_rate)
        # plot_efficient_frontier(returns, results, sdp, rp, sdp_min, rp_min)

        results, weights = simulate_random_portfolios2(returns, num_portfolios, risk_free_rate)
        plot_efficient_frontier2(close_prices_data, results, weights, merged_tickers_dict)

        st.markdown("#### Portfolio Optimization with Individual Stocks")
    
        st.info("""
        In this section, we take the actual **individual stock returns** from your selected dataset and apply advanced optimization techniques to construct the **best possible portfolio**.

        Rather than using pre-defined portfolios or ETFs, we build your investment mix from the ground up — choosing the **ideal combination of individual stocks** that either:

        - **Maximize the Sharpe Ratio** (best return per unit of risk), or  
        - **Minimize Volatility** (least amount of risk).

        This gives a **tailored investment strategy** based on real stock behavior, helping you understand how to make the most efficient use of your capital.
        """)

        an_vol, an_rt, returns, sdp, rp, sdp_min, rp_min, mean_returns, cov_matrix = \
            display_ef_with_selected(returns, tickers_dict, risk_free_rate)

        create_portfolio_optimization_plot(an_vol, an_rt, returns, sdp, rp, sdp_min, rp_min, mean_returns, cov_matrix, efficient_frontier)

        st.header("3. Data Preparation")
        # Checking for missing values and remove the missing values
        st.write("Check for missing values in the dataset")
        any_missing_values = close_prices_data.isnull().values.any()

        if any_missing_values:
            st.warning("⚠️ There are missing values in the dataset.")

            # Display a summary of missing values per column
            missing_summary = close_prices_data.isnull().sum()
            missing_summary = missing_summary[missing_summary > 0].reset_index()
            missing_summary.columns = ["Stock", "Missing Values"]
            st.markdown("### Missing Values Summary")
            st.dataframe(missing_summary)

            # Display the actual rows with missing values
            st.markdown("### Rows with Missing Values")
            st.dataframe(close_prices_data[close_prices_data.isnull().any(axis=1)])

            st.subheader("Data Cleaning")

            # First, fill missing values using forward fill (ffill) and backward fill (bfill).
            # This step uses nearby known values to fill in gaps, helping preserve columns with minor missing data.
            close_prices_data.ffill(inplace=True)
            close_prices_data.bfill(inplace=True)

            st.write("Used forward fill (ffill) and backward fill (bfill) to fill missing values. " \
            "This step uses nearby known values to fill in gaps, helping preserve columns with minor missing data.")

            # Then, drop any columns that still contain missing values after both fills.
            close_prices_data = drop_columns_with_excessive_missing_data(close_prices_data, threshold=0.3)

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

        # Get Minimum and maximum date
        # Use DataFrame index, because Date is index
        train_max_date = X_train.index.max()
        train_min_date = X_train.index.min()
        test_max_date = X_test.index.max()
        test_min_date = X_test.index.min()

        st.info(f"Train size: {train_len}, Start date: {train_min_date}, End date: {train_max_date}")
        st.info(f"Test size: {len(X) - train_len}, Start date: {test_min_date}, End date: {test_max_date}")

        # Calculate percentage return for training set
        returns = X_train.pct_change(fill_method=None).dropna()

        st.write("Displaying few Train returns:")
        st.write(returns.head())

        # Calculate percentage return for test set
        returns_test = X_test.pct_change(fill_method=None).dropna()

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
        condensed_dist = squareform(dist) # convert to condensed form
        link = linkage(condensed_dist, 'ward')

        # Show the first linkage step
        st.write("Showing the first linkage step")
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

        st.markdown("#### Portfolio Weights Allocation")
        portfolios_tmp = portfolios.round(4) # Round portfolio weights to 4 decimal places
        # portfolios_tmp['Name'] = portfolios_tmp.index.map(tickers_dict) # Add a column for company names based on ticker
        # Create Name column based on ticker mapping
        portfolios_tmp.insert(loc=0, column='Name', value=portfolios_tmp.index.map(tickers_dict))
        st.write(portfolios_tmp)

        # Visualize portfolios
        st.markdown("#### Portfolio Weights Allocation Piechart")
        create_portfolios_piechart(portfolios)

        # 6. Backtesting-Out Of Sample
        st.header("6. Backtesting-Out Of Sample")

        st.info("Take the portfolio weights, multiply them with the stock returns, and get the daily portfolio returns for both training and testing periods.")

        # Compute portfolio returns in-sample and out-of-sample
        st.subheader("6.1 In Sample and Out of Sample Results")

        # st.info("")

        def calculate_metrics(returns, risk_free_rate=0.0):
            """Compute annual return, volatility, and Sharpe ratio from daily returns."""
            mean_return = returns.mean() * 252  # Annualized return
            volatility = returns.std(axis=0) * np.sqrt(252)  # Annualized volatility
            sharpe = (mean_return - risk_free_rate) / volatility  # Sharpe ratio
            return mean_return, volatility, sharpe

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

        # Compute metrics for each portfolio
        rows = []
        for portfolio in ['MVP', 'HRP']:
            # Training metrics
            train_ret, train_vol, train_sharpe = calculate_metrics(in_sample_result[portfolio])
            # Test metrics
            test_ret, test_vol, test_sharpe = calculate_metrics(out_of_sample_result[portfolio])
            
            # Append a row to the summary table
            rows.append([
                portfolio,
                round(train_ret * 100, 2),
                round(train_vol * 100, 2),
                round(train_sharpe, 4),
                round(test_ret * 100, 2),
                round(test_vol * 100, 2),
                round(test_sharpe, 4)
            ])

        # Step 3: Build the summary DataFrame
        sample_summary_df = pd.DataFrame(rows, columns=[
            'Portfolio',
            'Training Return (%)',
            'Training Volatility (%)',
            'Training Sharpe',
            'Test Return (%)',
            'Test Volatility (%)',
            'Test Sharpe'
        ])

        st.info("Computed metrics for each portfolio")
        st.write(sample_summary_df)

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
