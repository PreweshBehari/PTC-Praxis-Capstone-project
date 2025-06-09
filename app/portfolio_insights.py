
##########################################################################################
# Imports
##########################################################################################
import streamlit as st
import plotly.express as px


##########################################################################################
# Prtfolio Insights Methods
##########################################################################################
def display_insights(in_sample, out_sample):

    # Add educational tooltip
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        **🔹 Standard Deviation (Risk):**  
        This measures how much the returns of a portfolio vary over time.  
        A **lower** standard deviation means more consistent performance (lower risk).

        **🔹 Sharpe Ratio:**  
        This indicates how much return you get per unit of risk.  
        A **higher** Sharpe ratio means better risk-adjusted performance.

        - Sharpe > 1 → Good  
        - Sharpe > 2 → Very good  
        - Sharpe > 3 → Excellent
        """)

    def analyze_performance(data, sample_label):
        st.markdown(f"#### 📊 {sample_label} Performance Analysis")

        # Extract values
        mvp_risk = data.loc["MVP", "stdev"]
        hrp_risk = data.loc["HRP", "stdev"]
        mvp_sharpe = data.loc["MVP", "sharp_ratio"]
        hrp_sharpe = data.loc["HRP", "sharp_ratio"]

        # Determine best strategy on each metric
        lower_risk_strategy = "MVP" if mvp_risk < hrp_risk else "HRP"
        better_return_strategy = "MVP" if mvp_sharpe > hrp_sharpe else "HRP"

        # Create two columns
        col1, col2 = st.columns(2)

        # Show metric values
        # Left column: Risk comparison
        with col1:
            st.write("##### 🔹 Risk (Standard Deviation)")
            st.write(f"- MVP: **{mvp_risk:.3f}** (**{mvp_risk:.1%}**)")
            st.write(f"- HRP: **{hrp_risk:.3f}** (**{hrp_risk:.1%}**)")
            st.write(f"✅ **Lower risk:** {lower_risk_strategy}")

        # Right column: Sharpe Ratio comparison
        with col2:
            st.write("##### 🔹 Sharpe Ratio (Risk-adjusted Return)")
            st.write(f"- MVP: **{mvp_sharpe:.3f}**")
            st.write(f"- HRP: **{hrp_sharpe:.3f}**")
            st.write(f"✅ **Higher Sharpe Ratio:** {better_return_strategy}")

        # Verdict for this sample
        output = None
        if lower_risk_strategy == better_return_strategy:
            st.success(f"🏆 Overall winner: **{lower_risk_strategy}** (better in both risk and return)")
            output = lower_risk_strategy
        else:
            st.info(f"⚖️ It's a trade-off: **{lower_risk_strategy}** has lower risk, but **{better_return_strategy}** has better return")
            output = None  # Mixed result

        # Bar charts
        data_reset = data.reset_index().rename(columns={"index": "Strategy"})
        st.write("#### 📈 Visual Comparison")

        # Create two columns
        col1, col2 = st.columns(2)

        # Risk (Standard Deviation) chart in left column
        with col1:
            risk_fig = px.bar(
                data_reset, x="Strategy", y="stdev",
                title="Risk (Standard Deviation)", color="Strategy",
                labels={"stdev": "Standard Deviation"}
            )
            st.plotly_chart(risk_fig, use_container_width=True)

        # Sharpe Ratio chart in right column
        with col2:
            sharpe_fig = px.bar(
                data_reset, x="Strategy", y="sharp_ratio",
                title="Sharpe Ratio (Risk-Adjusted Return)", color="Strategy",
                labels={"sharp_ratio": "Sharpe Ratio"}
            )
            st.plotly_chart(sharpe_fig, use_container_width=True)

        return output

    # Show results
    # st.header("📋 Strategy Performance Summary")

    st.markdown("#### 6.2.1 In Sample Results")
    # st.dataframe(in_sample)
    analyze_performance(in_sample, "In Sample")

    st.markdown("#### 6.2.2 Out of Sample Results")
    # st.dataframe(out_sample)
    best_out_sample = analyze_performance(out_sample, "Out of Sample")

    # Final recommendation
    st.markdown("#### 6.2.3 Final Recommendation")
    if best_out_sample:
        st.success(f"✅ Based on Out-of-Sample performance, we recommend using the **{best_out_sample}** strategy — it has better risk-adjusted performance for real-world scenarios.")
    else:
        st.warning("⚖️ Both strategies have trade-offs in the Out-of-Sample test. You may prefer **HRP** for slightly lower risk or **MVP** for better returns with a slightly higher risk.")


def recommend_stocks(weights_df, strategy_name="HRP", threshold=0.005, tickers={}):

    if strategy_name not in weights_df.columns:
        st.error(f"Strategy '{strategy_name}' not found in weights.")
        return

    # Extract weights for the selected strategy
    strategy_weights = weights_df[strategy_name]

    # Filter out stocks with zero or near-zero allocation
    selected = strategy_weights[strategy_weights > threshold].sort_values(ascending=False)

    if selected.empty:
        st.warning("No stocks received significant allocation in this portfolio.")
        return

    st.write(f"You can invest in the following {len(selected)} stocks under the **{strategy_name}** strategy:")

    # Convert to DataFrame and reset index
    df = selected.rename("Allocation").to_frame().reset_index()

    # Rename the column that was previously the index (usually called 0 or the ticker code)
    df.columns = ["Ticker", "Allocation"]

    # Add Stock Name column
    df["Stock Name"] = df["Ticker"].map(lambda ticker: tickers.get(ticker, ticker))

    # Reorder columns
    df = df[["Ticker", "Stock Name", "Allocation"]]

    # Display without index
    st.dataframe(df.style.hide(axis="index").format({"Allocation": "{:.2%}"}), hide_index=True)

    # # Human summary
    # st.markdown("##### Summary")
    # st.write(f"You can invest in the following {len(selected)} stocks under the **{strategy_name}** strategy:")
    # for stock, weight in selected.items():
    #     stock_name = tickers.get(stock, stock) # fallback to ticker if name not found
    #     st.write(f"- **{stock_name}**: {weight:.2%} of your portfolio")

    # total_allocation = selected.sum()
    # if total_allocation < 0.99:
    #     st.info(f"⚠️ Total allocation is only {total_allocation:.2%}. Consider reviewing the model or adding more investable assets.")
