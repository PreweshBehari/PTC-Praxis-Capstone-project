# 📈 PTC Praxis Capstone Project: Building Efficient Stock Portfolios

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Project Features](#2-project-features)
- [3. Project Structure](#3-project-structure)
  - [3.1 App Module Explained](#31-app-module-explained)
    - [3.1.1 data_download.py](#311-data_downloadpy)
    - [3.1.2 data_preparation.py](#312-data_preparationpy)
    - [3.1.3 data_transformation.py](#313-data_transformationpy)
    - [3.1.4 data_visualization.py](#314-data_visualizationpy)
    - [3.1.5 optimizer.py](#315-optimizerpy)
    - [3.1.6 portfolio_insights.py](#316-portfolio_insightspy)
    - [3.1.7 utils.py](#317-utilspy)
- [4. Local Setup](#4-local-setup)
  - [4.1 Prerequisites](#41-prerequisites)
  - [4.2 Clone the Repository](#42-clone-the-repository)
  - [4.3 Create conda environment](#43-create-conda-environment)
  - [4.4 Install Dependencies](#44-install-dependencies)
  - [4.5 Running the Web Application](#45-running-the-web-application)

## 1. Introduction

This project focuses on downloading, cleaning, transforming, and visualizing financial data (specifically S&P 500 stocks) using **Streamlit** for interactivity and **Plotly/Matplotlib** for visualization.

It supports side-by-side analysis of two portfolio construction methods:
- **Markowitz Mean-Variance Optimization**
- **Hierarchical Risk Parity (HRP)**

The goal is to evaluate how traditional optimization compares to a more modern, machine-learning-based clustering approach when applied to historical data from S&P 500 stocks. The project includes a user-friendly **Streamlit web application** for interactive portfolio analysis and visualization.

---

## 2. Project Features

- **Historical Stock Data Analysis** via `yfinance`
- **Two Optimization Methods**:
  - *Markowitz* (using `PyPortfolioOpt`)
  - *HRP* (custom implementation based on hierarchical clustering)
- **Interactive Visualizations**: correlation heatmaps, dendrograms, efficient frontiers
- **Insight Generation**: risk-return comparisons, weight distributions
- **Web Interface**: built using `Streamlit` for ease of use

---

## 3. Project Structure

```
├── app
│   ├── __init__.py
│   ├── data_download.py          # Downloads S&P 500 stock data using yfinance
│   ├── data_preparation.py       # Cleans and structures the data
│   ├── data_transformation.py    # Applies returns and risk calculations
│   ├── data_visualization.py     # Contains Streamlit plots and chart logic
│   ├── optimizer.py              # Runs Markowitz and HRP optimization
│   ├── portfolio_insights.py     # Generates insights and reports
│   ├── utils.py                  # Shared helper functions
│   ├── data
│   │   ├── stock_prices_[hash].csv  # Historical price data
│   │   ├── tickers.json             # List of S&P 500 tickers
├── app.py                        # Main Streamlit application entry point
├── requirements.txt              # Python dependencies
```

### 3.1 App Module Explained

#### 3.1.1 data_download.py

**Purpose:**
- Retrieve and store S&P 500 stock tickers, and download their historical close prices using Yahoo Finance (`yfinance`).

**Functions:**
- `get_tickers(save_path="data/tickers.json") -> dict`
  - Retrieves S&P 500 ticker symbols and company names from Wikipedia.
  - Converts them for Yahoo Finance compatibility (e.g., `BRK.B` → `BRK-B`).
  - Caches results to a local JSON file to reduce repeated downloads.
  - Returns: `{ticker: company_name}` dictionary.

- `download_close_prices(tickers, start_date="2020-01-01", end_date=None, ticker_names={}) -> pd.DataFrame`
  - Downloads 'Close' prices for a list of tickers via `yfinance`.
  - Uses Streamlit to show progress and status updates.
  - Returns: DataFrame with date index and ticker symbols as columns.


#### 3.1.2 data_preparation.py

**Purpose:**
- Provides a utility function to clean the dataset by removing columns with excessive missing values.

**Function:**
- `drop_columns_with_excessive_missing_data(dataset, threshold=0.3) -> pd.DataFrame`
  - Drops columns from the dataset where the percentage of missing values exceeds the threshold (default: 30%).
  - Uses Streamlit to display columns dropped and the resulting shape of the dataset.

#### 3.1.3 data_transformation.py

**Purpose:**
- Prepares the dataset for time series modeling by splitting it into training and test sets.

**Function:**
- `split_dataset(dataset, train_fraction=0.8) -> Tuple[pd.DataFrame, int]`
  - Returns a deep copy of the dataset and the number of rows to use for training.
  - `train_fraction` controls the proportion of data reserved for training (default: 80%). 

#### 3.1.4 data_visualization.py

**Purpose:**
- Generates visualizations for data understanding and portfolio analysis using Plotly and Matplotlib.

**Functions:**
- `create_heatmap(df, topn=10)`
  - Displays a correlation heatmap of the top N most volatile stocks.
  - Uses `plotly.express.imshow()` with Streamlit rendering.
- `create_dendrogram(link, X)`
  - Plots a hierarchical clustering dendrogram from a linkage matrix and data columns.
  - Useful for HRP (Hierarchical Risk Parity) interpretation.
- `create_portfolios_piechart(portfolios)`
  - Shows portfolio allocation as two pie charts:
    - MVP (Minimum Variance Portfolio)
    - HRP (Hierarchical Risk Parity)
  - Assumes portfolios has 2 columns representing weights for each method.
- `create_in_and_out_sample_plots(in_sample_result, out_of_sample_result)`
  - Plots cumulative returns for both in-sample and out-of-sample periods.
  - Visual comparison of portfolio performance over time. 

#### 3.1.5 optimizer.py

**Purpose:**
- Compute optimal portfolio allocations based on financial asset return data using quantitative portfolio optimization techniques. Specifically, it implements two prominent strategies:
  1. Minimum Variance Portfolio (MVP) – based on mean-variance optimization (Markowitz framework).
  2. Hierarchical Risk Parity (HRP) – based on hierarchical clustering and risk allocation.


`optimizer.py` helps determine how much weight to assign to each asset in a portfolio to optimize for **risk-adjusted returns**.
It does this using mathematical and statistical tools including:
- Covariance/correlation matrix analysis
- Hierarchical clustering
- Convex optimization (quadratic programming)

**Functions:**
- `correlDist(corr)`
  - Converts a correlation matrix to a distance matrix used for clustering in HRP.

- `getQuasiDiag(link)`
  - Sorts clustered items to ensure similar assets are placed adjacently in the dendrogram order.

- `getIVP(cov)`
  - Returns asset weights inversely proportional to their variance.

- `getClusterVar(cov, cItems)`
  - Computes the total variance of a cluster using IVP weights.

- `getRecBipart(cov, sortIx)`
  - Recursively partitions asset clusters and assigns weights using HRP logic.

- `getMVP(cov)`
  - Computes MVP weights by solving a quadratic optimization problem. Uses `cvxopt` for efficient frontier optimization and selects the optimal portfolio based on Sharpe-like efficiency.

- `getHRP(cov, corr)`
  - Combines hierarchical clustering and recursive bisection to calculate HRP weights. Handles linkage computation and distance transformation internally.

- `build_portfolios(returns)`
  - Builds both MVP and HRP portfolios from asset return data.
  - Returns: A DataFrame with assets as rows and optimization strategies (`MVP`, `HRP`) as columns.

#### 3.1.6 portfolio_insights.py

**Purpose:**
Provides Streamlit-powered user interface to:
- Compare MVP vs. HRP strategies using risk and Sharpe ratio
- Display visual insights and bar charts
- Recommend stocks to invest in based on chosen strategy 

**Functions:**
- `display_insights(in_sample, out_sample)`
  - Presents side-by-side analysis of the two strategies using tables, bar charts, and performance metrics like risk (std dev) and Sharpe ratio. Highlights which strategy performs better.

- `recommend_stocks(weights_df, strategy_name="HRP", threshold=0.005, tickers={})`
  - Lists stocks recommended for investment based on portfolio weights from the selected strategy. Includes a table and a human-readable summary.

> Includes educational tooltips and visual components using Plotly and Streamlit widgets.

#### 3.1.7 utils.py

**Purpose:**
- Provides utility functions for hashing and formatting.

**Functions:**
- `create_unique_value(combined_keys)`
  - Returns a consistent MD5 hash from a string input (used for caching, keys, etc.).

- `human_readable_date(input_date, input_date_format="%Y-%m-%d", output_date_format="%d %B %Y")`
  - Converts a date string or datetime object into a user-friendly string (e.g., "2025-05-10" → "10 May 2025"). 

---

## 4. Local Setup

### 4.1 Prerequisites

Ensure the following are installed:

- [Python 3.12+](https://www.python.org/downloads/)
- [Anaconda](https://www.anaconda.com/download)

### 4.2 Clone the Repository

```bash
git clone https://github.com/PreweshBehari/PTC-Praxis-Capstone-project.git
```

#### 4.3 Create conda environment

In your VSCode workspace/Explorer section, right-click and choose the option `Open in Integrated Terminal`.

In the terminal, execute the following commands:
```bash
conda create --name ptc-praxis-capstone python=3.12
conda activate ptc-praxis-capstone
```

#### 4.4 Install Dependencies

This project uses the following dependencies:

```bash
streamlit==1.45.0
pandas==2.2.3
numpy==2.2.5
scikit-learn==1.6.1
scipy==1.15.2
matplotlib==3.10.1
seaborn==0.13.2
pyportfolioopt==1.5.6
yfinance==0.2.58
lxml==5.4.0
cvxopt==1.3.2
```

Install the dependencies:

```bash   
pip install -r requirements.txt
```

#### 4.5 Running the Web Application

Once set up, launch the app using:

```bash
streamlit run app.py
```

Your default browser will open an interactive dashboard for exploring portfolio construction and results.

