##########################################################################################
# Imports
##########################################################################################
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns


def get_efficient_portfolio(price_df, tickers_dict, topn=30):
    """
    Given a DataFrame of historical prices (columns=tickers), compute the optimal top N stock
    portfolio using the Efficient Frontier (Max Sharpe ratio strategy).

    Args:
        price_df (pd.DataFrame): Historical close prices.
        topn (int): Number of top stocks to select based on initial weight.

    Returns:
        dict: Dictionary of selected tickers and their portfolio weights.
    """
    # Drop tickers with too many missing values
    price_df = price_df.dropna(axis=1, thresh=int(0.8 * len(price_df)))

    # Calculate expected returns and sample covariance
    mean_return = expected_returns.mean_historical_return(price_df)
    sample_cov = risk_models.sample_cov(price_df)

    # Step 1: Run Efficient Frontier on full data
    ef = EfficientFrontier(mean_return, sample_cov)
    weights = ef.max_sharpe()
    weights_series = pd.Series(weights).sort_values(ascending=False)

    # Step 2: Select top N tickers by initial weight
    top_tickers = weights_series.head(topn).index.tolist()
    topn_names = {ticker: tickers_dict[ticker] for ticker in top_tickers}

    # Step 3: Recalculate Efficient Frontier for top N
    top_price_df = price_df[top_tickers]
    mean_return_top = expected_returns.mean_historical_return(top_price_df)
    sample_cov_top = risk_models.sample_cov(top_price_df)

    ef_top = EfficientFrontier(mean_return_top, sample_cov_top)
    top_weights = ef_top.max_sharpe()
    cleaned_weights = ef_top.clean_weights()

    # Attach ticker names to final result
    portfolio = {ticker: {
        "name": topn_names[ticker],
        "weight": round(weight, 4)
    } for ticker, weight in cleaned_weights.items() if weight > 0}

    return portfolio