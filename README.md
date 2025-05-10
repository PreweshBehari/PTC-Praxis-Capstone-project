# 📈 PTC Praxis Capstone Project: Building Efficient Stock Portfolios

## Table of Contents
- [1. Introduction](#1-introduction)
- [2. Project Features](#2-project-features)
- [3. Project Structure](#3-project-structure)
- [4. Local Setup](#4-local-setup)
  - [4.1 Prerequisites](#41-prerequisites)
  - [4.2 Clone the Repository](#42-clone-the-repository)
  - [4.3 Create conda environment](#43-create-conda-environment)
  - [4.4 Install Dependencies](#44-install-dependencies)
  - [4.5 Running the Web Application](#45-running-the-web-application)

## 1. Introduction

This capstone project explores and compares two advanced methods for stock portfolio optimization:

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

