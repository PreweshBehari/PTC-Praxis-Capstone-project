##########################################################################################
# Imports
##########################################################################################
import numpy as np
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
    top_cols = df[numeric_cols].std().sort_values(ascending=False).head(topn).index
    corr_matrix = df[top_cols].corr()

    # Create heatmap with annotations
    fig = px.imshow(
        corr_matrix,
        title="Correlation Heatmap",
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

    # Efficient frontier
    ax.plot(risks, rets, 'g--', label='Efficient Frontier')

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
        ax.plot(returns.index, returns[c], lw=3, alpha=0.8, label=tickers_dict.get(c, c))

    # Customize the plot
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylabel('Daily Returns')

    # Render in Streamlit
    st.pyplot(fig)