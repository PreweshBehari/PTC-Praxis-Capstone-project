##########################################################################################
# Imports
##########################################################################################
import streamlit as st
import plotly.express as px

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


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

    # Stretch the plot to fill the window
    fig.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),  # optional: reduce whitespace
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