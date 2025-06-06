##########################################################################################
# Imports
##########################################################################################
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Package for optimization of mean variance optimization
import cvxopt as opt
from cvxopt import blas, solvers

##########################################################################################
# Data Visualization Methods
##########################################################################################

def create_heatmap(df, topn=10):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Focus only on top N most-volatile stocks
    top_cols = df[numeric_cols].std(axis=0).sort_values(ascending=False).head(topn).index
    corr_matrix = df[top_cols].corr()

    # Create heatmap with annotations
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",  # Annotate with values rounded to 2 decimals
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1
    )

    # Increase the size of the plot
    fig.update_layout(
        width=900,  # Adjust width
        height=900,  # Adjust height
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def create_cov_heatmap(cov_matrix):
    # Create heatmap with annotations
    fig = px.imshow(
        cov_matrix,
        text_auto=".6f",  # Annotate with values rounded to 6 decimals
        color_continuous_scale="YlGnBu",
    )

    # Increase the size of the plot
    fig.update_layout(
        width=900,  # Adjust width
        height=900,  # Adjust height
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def create_dendrogram(link, X):
    # Plot Dendrogram
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_title("Dendrograms")
    dendrogram(link, labels=X.columns, ax=ax)

    # Display in Streamlit
    st.pyplot(fig)

def create_portfolios_piechart(portfolios):
    # Create a 1x2 subplot figure for MVP and HRP pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))

    # Plot pie chart for Minimum Variance Portfolio (MVP)
    ax1.pie(portfolios.iloc[:, 0], labels=portfolios.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('MVP', fontsize=30)

    # Plot pie chart for Hierarchical Risk Parity Portfolio (HRP)
    ax2.pie(portfolios.iloc[:, 1], labels=portfolios.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('HRP', fontsize=30)

    # Display the figure in Streamlit
    st.pyplot(fig)

def create_in_and_out_sample_plots(in_sample_result, out_of_sample_result):
    # Plotting In-sample cumulative returns
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    in_sample_result.cumsum().plot(ax=ax1, title="In-Sample Results")
    st.pyplot(fig1)

    # Plotting Out-of-sample cumulative returns
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    out_of_sample_result.cumsum().plot(ax=ax2, title="Out Of Sample Results")
    st.pyplot(fig2)

def create_efficient_frontier_plot(risks, returns):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(risks, returns, 'g--', label='Efficient Frontier')
    ax.set_xlabel('Risk')
    ax.set_ylabel('Returns')
    ax.set_title('Efficient Frontier')
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

def plot_efficient_frontier(returns, results, sdp, rp, sdp_min, rp_min):
    cov = returns.cov().values
    n = len(cov)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    S = opt.matrix(cov)
    pbar = opt.matrix(returns.mean().values)
    # pbar = opt.matrix(np.ones(cov.shape[0]))

    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    rets = [blas.dot(pbar, x) for x in portfolios]

    # Plotting in Streamlit
    fig, ax = plt.subplots(figsize=(10, 7))

    # Simulated portfolios
    sc = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    # # Efficient frontier
    # ax.plot(risks, rets, 'g--', label='Efficient Frontier')

    # Special portfolios
    ax.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
    ax.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum Volatility')

    ax.set_title('Simulated Portfolio Optimization based on Efficient Frontier')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Returns')
    ax.legend(labelspacing=0.8)
    ax.grid(True)

    st.pyplot(fig)

def plot_efficient_frontier2(table, results, weights, tickers_dict):
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    st.markdown("##### Maximum Sharpe Ratio Portfolio Allocation")

    st.info(f"""
    This portfolio is designed to give you the best possible return for the amount of risk you take.  
    It balances **how much you expect to earn** against **how much the investment values might go up and down**.

    - The **Sharpe Ratio** measures this balance: higher values mean better returns for each unit of risk.  
    - This portfolio picks weights for each asset to maximize that ratio.  

    Here are the key figures for this portfolio:  
    - **Annualised Return:** {rp:.2%} (expected yearly return)  
    - **Annualised Volatility:** {sdp:.2%} (expected yearly fluctuations in return)
    """)

    # Convert the DataFrame from wide format (stocks as columns) to long format (stocks as rows with allocation as a value)
    long_max_sharpe_allocation = max_sharpe_allocation.T.reset_index()
    long_max_sharpe_allocation.columns = ['Stock', 'Allocation']
    # Insert 'Name' column before 'Allocation'
    long_max_sharpe_allocation.insert(
        loc=1,  # position index to insert before 'Allocation'
        column='Name',
        value=long_max_sharpe_allocation['Stock'].map(tickers_dict)
    )

    st.write(long_max_sharpe_allocation.sort_values(by='Allocation', ascending=False).reset_index(drop=True))
    
    st.markdown("##### Minimum Volatility Portfolio Allocation")

    st.info(f"""
    This portfolio focuses on **minimizing risk**, rather than maximizing returns.  
    It allocates weights to assets in a way that keeps the overall **fluctuations (volatility) as low as possible**.

    This is a good choice for more conservative investors or those who prefer **steady and stable performance** over time.

    Here are the key figures for this portfolio:  
    - **Annualised Return:** {rp_min:.2%} (expected yearly return)  
    - **Annualised Volatility:** {sdp_min:.2%} (expected yearly fluctuations in return)
    """)

    # Convert the DataFrame from wide format (stocks as columns) to long format (stocks as rows with allocation as a value)
    long_min_vol_allocation= min_vol_allocation.T.reset_index()
    long_min_vol_allocation.columns = ['Stock', 'Allocation']
    # Insert 'Name' column before 'Allocation'
    long_min_vol_allocation.insert(
        loc=1,  # position index to insert before 'Allocation'
        column='Name',
        value=long_min_vol_allocation['Stock'].map(tickers_dict)
    )
    st.write(long_min_vol_allocation.sort_values(by='Allocation', ascending=False).reset_index(drop=True))

    st.markdown("#### Simulated Portfolio Optimization based on Efficient Frontier")

    # Plotting in Streamlit
    fig, ax = plt.subplots(figsize=(10, 7))

    # Simulated portfolios
    sc = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Sharpe Ratio')

    # Special portfolios
    ax.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')
    ax.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum Volatility')

    ax.set_title('Simulated Portfolio Optimization based on Efficient Frontier')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Returns')
    ax.legend(labelspacing=0.8)
    ax.grid(True)

    st.pyplot(fig)

def create_stock_price_evolved_plot(table, tickers_dict):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each column in the DataFrame
    for c in table.columns.values:
        ax.plot(table.index, table[c], lw=3, alpha=0.8, label=tickers_dict.get(c, c))

    # Customize labels and legend
    ax.set_ylabel('Price in $')
    ax.legend(loc='upper left', fontsize=8)

    # Display the figure in Streamlit
    st.pyplot(fig)

def create_daily_returns_plot(returns, tickers_dict):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot each time series
    for c in returns.columns.values:
        ax.plot(returns.index, returns[c], lw=3, alpha=0.85, label=tickers_dict.get(c, c))

    # Customize the plot
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylabel('Daily Returns')

    # Render in Streamlit
    st.pyplot(fig)

def create_portfolio_optimization_plot(an_vol, an_rt, returns, sdp, rp, sdp_min, rp_min, mean_returns, cov_matrix, efficient_frontier):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot of individual stock portfolios
    ax.scatter(an_vol, an_rt, marker='o', s=200)

    # Add annotations for each stock
    for i, txt in enumerate(returns.columns):
        ax.annotate(txt, (an_vol.iloc[i], an_rt.iloc[i]), xytext=(10, 0), textcoords='offset points')

    # Max Sharpe ratio portfolio
    ax.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe Ratio')

    # Minimum volatility portfolio
    ax.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum Volatility')

    # Efficient frontier
    target = np.linspace(rp_min, 0.8, 50)
    # target = np.linspace(rp_min, 0.34, 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='Efficient Frontier')

    # Labels and styling
    ax.set_title('Portfolio Optimization with Individual Stocks')
    ax.set_xlabel('Annualized Volatility (Risk)')
    ax.set_ylabel('Annualized Returns')
    ax.legend(labelspacing=0.8)
    ax.grid(True)

    st.pyplot(fig)