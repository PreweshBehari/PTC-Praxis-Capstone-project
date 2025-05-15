##########################################################################################
# Imports
##########################################################################################
# Load libraries
import numpy as np
import pandas as pd
import streamlit as st

# Import Model Packages
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

# Package for optimization of mean variance optimization
import cvxopt as opt
from cvxopt import blas, solvers

import scipy.optimize as sco

##########################################################################################
# Optimizing Methods
##########################################################################################
def correlDist(corr):
    """
    Convert a correlation matrix into a distance matrix for clustering.

    The distance is computed such that:
      - A perfect positive correlation (corr = 1) results in distance 0.
      - A perfect negative correlation (corr = -1) results in distance 1.
      - A correlation of 0 results in distance sqrt(0.5) ≈ 0.707.

    This distance matrix is used in hierarchical clustering algorithms (e.g., for HRP portfolios).

    Parameters:
      corr (pd.DataFrame or np.ndarray): Correlation matrix of asset returns.

    Returns:
      pd.DataFrame or np.ndarray: Symmetric distance matrix suitable for clustering.
    """

    # Compute the distance matrix from the correlation matrix
    # Formula: distance = sqrt((1 - correlation) / 2)
    # Ensures the distance is between 0 and 1 (0 <= d[i,j] <= 1)
    dist = ((1 - corr) / 2.) ** 0.5 # distance matrix

    return dist

##########################################################################################
# HRP Optimizing Methods
##########################################################################################

def getQuasiDiag(link):
    """
    Given a hierarchical clustering linkage matrix,
    this function sorts the clustered items into a quasi-diagonalized order.

    The goal is to rearrange the assets so that assets that are close (more correlated)
    are grouped together, following the hierarchical structure.

    Args:
        link (ndarray): Linkage matrix from hierarchical clustering.

    Returns:
        list: Ordered list of asset indices based on hierarchical clustering.
    """

    # Ensure all entries in the linkage matrix are integers
    link = link.astype(int)

    # Initialize the sort index with the two clusters merged in the final step
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])

    # Total number of original items (assets)
    numItems = link[-1, 3]

    # Continue expanding the sort index until only original items (no clusters) are left
    while sortIx.max() >= numItems:

        # Expand the index to make space for new items
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)

        # Find elements that are clusters (not original items)
        df0 = sortIx[sortIx >= numItems]

        # Indices where clusters were found
        i = df0.index

        # Cluster indices in the linkage matrix
        j = df0.values - numItems

        # Replace cluster with its two children (first child)
        sortIx[i] = link[j, 0]

        # Insert second child just after the first
        df0 = pd.Series(link[j, 1], index=i + 1)
        #sortIx = sortIx.append(df0) # Deprecated: replace with pd.concat()
        sortIx = pd.concat([sortIx, df0]) # Concatenate instead of append

        # Sort index to maintain correct order
        sortIx = sortIx.sort_index()

        # Reset index
        sortIx.index = range(sortIx.shape[0])

    # Return final sorted list of asset indices
    return sortIx.tolist()

def getIVP(cov, **kargs):
    """
    Recursive bisection: Compute the Inverse-Variance Portfolio (IVP) weights.

    In an IVP, assets with lower variance (more stable) get higher weights.
    This method assumes no correlation between assets (only variance matters).

    Args:
        cov (pd.DataFrame or np.ndarray): Covariance matrix of asset returns.
        **kargs: Extra keyword arguments (not used here, but kept for flexibility).

    Returns:
        np.ndarray: Weights for the assets following the IVP strategy.
    """
    # 1. Get variances (diagonal elements of the covariance matrix)
    ivp = 1. / np.diag(cov)

    # 2. Normalize the weights to sum to 1
    ivp /= ivp.sum()

    return ivp

def getClusterVar(cov, cItems):
    """
    Compute the variance of a cluster of assets using the Inverse-Variance Portfolio weights.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        cItems (list): List of asset indices in the cluster.

    Returns:
        float: Variance of the cluster.
    """
    # 1. Slice the covariance matrix to only include assets in the cluster
    cov_ = cov.loc[cItems, cItems]

    # 2. Get IVP weights for the assets in the cluster
    w_ = getIVP(cov_).reshape(-1, 1) # Reshape for matrix multiplication

    # 3. Compute the variance: w' * cov * w
    cVar = np.dot(np.dot(w_.T, cov_), w_)[0, 0]

    return cVar

def getRecBipart(cov, sortIx):
    """
    Recursively assign weights to assets based on hierarchical clustering (HRP method).

    This function splits the sorted list of assets into two halves at each level,
    and allocates weights between them based on their cluster variances.

    Args:
        cov (pd.DataFrame): Covariance matrix of asset returns.
        sortIx (list): Ordered list of asset indices (from getQuasiDiag).

    Returns:
        pd.Series: HRP portfolio weights for each asset.
    """
    # 1. Initialize all weights as 1
    w = pd.Series(1, index=sortIx)

    # 2. Start with all assets grouped into one cluster
    cItems = [sortIx]

    # 3. Recursively bisect the clusters until single assets
    while len(cItems) > 0:
        # Bisect each cluster into two halves
        cItems = [i[j:k] for i in cItems for j, k in (
            (0, len(i) // 2), (len(i) // 2, len(i))
        ) if len(i) > 1]

        # 4. Iterate through clusters in pairs
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i]     # First cluster
            cItems1 = cItems[i + 1] # Second cluster

            # 5. Calculate variances for each sub-cluster
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)

            # 6. Allocate weights inversely proportional to cluster variances
            alpha = 1 - cVar0 / (cVar0 + cVar1)

            # Convert to float to avoid dtype warnings
            w = w.astype(float)

            # 7. Adjust weights based on the allocation ratio
            w[cItems0] *= alpha
            w[cItems1] *= 1 - alpha

    return w

##########################################################################################
# General Optimizing Methods
##########################################################################################

def getMVP(returns):
    """
    Calculate the Minimum Variance Portfolio (MVP) weights using quadratic programming.

    This function computes the efficient frontier and determines the minimum variance portfolio 
    using the covariance matrix of asset returns. It applies convex optimization (quadratic programming)
    to solve for portfolio weights under the constraint that all weights sum to 1 and are non-negative 
    (i.e., no short-selling).

    Parameters:
    -----------
    returns : pandas.DataFrame
        A DataFrame where each column represents the historical returns of a different asset.

    Returns:
    --------
    list
        A list of optimal weights for each asset in the portfolio, representing the Minimum Variance Portfolio.
    
    Notes:
    ------
    - Uses the `cvxopt` library for solving the quadratic programming problem.
    - Assumes equal expected returns for all assets (as a simplification).
    - No short selling is allowed (weights >= 0).
    """
    cov = returns.cov().T.values
    n = len(cov)
    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(cov)
    pbar = opt.matrix(np.ones(cov.shape[0]))  # Assumes equal expected return for all assets

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

    # Calculate risks and returns for frontier
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Fit a 2nd degree polynomial to the frontier curve
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # Calculate the optimal minimum variance portfolio
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return list(wt)

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    dist = correlDist(corr)
    # Use scipy.spatial.distance.squareform() to convert the square distance matrix to the condensed form.
    condensed_dist = squareform(dist.values)
    link = sch.linkage(condensed_dist, 'single')
    #plt.figure(figsize=(20, 10))
    #dn = sch.dendrogram(link, labels=cov.index.values)
    #plt.show()
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()

def build_portfolios(returns):
    """
    Calculate portfolio allocations using both Minimum Variance Portfolio (MVP) and Hierarchical Risk Parity (HRP).

    Args:
        returns (pd.DataFrame): A dataframe containing asset returns (e.g., daily returns).

    Returns:
        pd.DataFrame: A dataframe where:
            - Rows represent asset names.
            - Columns represent portfolio strategies (MVP and HRP).
            - Each value represents the weight allocated to the asset in that strategy.

    Steps:
        1. Calculate the covariance and correlation matrices of the asset returns.
        2. Calculate HRP weights using hierarchical clustering.
        3. Calculate MVP weights using mean-variance optimization.
        4. Format both results into a single dataframe.
    """
    # Step 1: Calculate covariance and correlation matrices
    cov, corr = returns.cov(), returns.corr()

    # Step 2: Compute the HRP portfolio weights
    hrp = getHRP(cov, corr)

    # Step 3: Compute the MVP portfolio weights
    mvp = getMVP(returns)

    # Step 4: Convert MVP weights to a Pandas Series with asset names as index
    mvp = pd.Series(mvp, index=cov.index)

    # Step 5: Combine both portfolios into a DataFrame
    portfolios = pd.DataFrame([mvp, hrp], index=['MVP', 'HRP']).T

    # Alternative (commented out): could replace MVP with IVP (Inverse Variance Portfolio) if needed
    # portfolios = pd.DataFrame([ivp, hrp], index=['IVP', 'HRP']).T

    return portfolios

def calculate_sample_metrics(returns, trading_days=252):
    """
    Calculate and return annualized standard deviation (risk) and Sharpe ratio
    for in/out - sample portfolio returns.

    Args:
        returns (pd.DataFrame): In/Out sample portfolio returns for MVP and HRP.
        trading_days (int): Number of trading days in a year (default is 252).

    Returns:
        pd.DataFrame: A DataFrame containing annualized standard deviation and Sharpe ratio
                      for each portfolio strategy.
    """
    # Annualized standard deviation (volatility)
    stddev = returns.std() * np.sqrt(trading_days)

    # Sharpe Ratio: Annualized return divided by standard deviation
    sharp_ratio = (returns.mean() * np.sqrt(trading_days)) / (returns.std())

    # Create a results table
    results = pd.DataFrame(dict(stdev=stddev, sharp_ratio=sharp_ratio))

    return results

def simulate_random_portfolios(returns, num_portfolios=5000, risk_free_rate=0.0, trading_days=252):
    np.random.seed(42)
    mean_returns = returns.mean() * trading_days  # Annualize
    cov_matrix = returns.cov() * trading_days     # Annualize

    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_std_dev # volatility
        results[1, i] = portfolio_return  # return
        results[2, i] = sharpe_ratio      # sharpe

    # Identify max Sharpe and min Volatility
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]

    return results, sdp, rp, sdp_min, rp_min

def portfolio_annualised_performance(weights, mean_returns, cov_matrix, trading_days=252):
    returns = np.sum(mean_returns*weights ) * trading_days
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(trading_days)
    return std, returns

def simulate_random_portfolios2(returns, num_portfolios=5000, risk_free_rate=0.0):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns) # number of assets
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weights_record





def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_ef_with_selected(returns, tickers_dict, risk_free_rate=0.0):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=returns.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    max_sharpe_allocation

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x,index=returns.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252

    st.markdown("##### Maximum Sharpe Ratio Portfolio Allocation")
    st.write(f"Annualised Return: {round(rp, 2)}")
    st.write(f"Annualised Volatility: {round(sdp,2)}")

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
    st.write(f"Annualised Return: {round(rp_min, 2)}")
    st.write(f"Annualised Volatility: {round(sdp_min,2)}")

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

    st.markdown("##### Individual Stock Returns and Volatility")

    # Create a list of dictionaries to hold the data
    data = []
    for i, txt in enumerate(returns.columns):
        data.append({
            "Stock": txt,
            "Name": tickers_dict.get(txt, txt),
            "Annualised Return": round(an_rt.iloc[i], 2),
            "Annualised Volatility": round(an_vol.iloc[i], 2)
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(data)

    # Show in Streamlit
    st.dataframe(summary_df.sort_values(by="Annualised Return", ascending=False).reset_index(drop=True))

    return an_vol, an_rt, returns, sdp, rp, sdp_min, rp_min, mean_returns, cov_matrix