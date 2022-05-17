# THE PORTFOLIO SELECTION ALGRORITHM
import copy
import numpy as np
import pandas as pd
import requests
import io
from pypfopt import plotting
import logging
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import pandas_datareader.data as web
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
from pypfopt.black_litterman import BlackLittermanModel
"""CHECK SCRIPTS IN THIS LIBRARY
https://pypi.org/project/pyportfolioopt/#an-overview-of-classical-portfolio-optimization-methods
https://builtin.com/data-science/portfolio-optimization-python
https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
"""

# LOGGER
log = logging.getLogger(__name__)
log.debug

START = datetime(2021, 1, 1)
END = datetime(2022, 1, 1)

# PUT NEW CLASSES HERE
class Stocks:
    def __init__(self, df=pd.DataFrame):
        self.df = df

# PUT NEW FUNCTIONS HERE
def make_portfolio_mvo(df = pd.DataFrame):
    try:
        # Calculate expected returns and sample covariance
        mu = mean_historical_return(df)
        S = sample_cov(df)
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        raw_weights_mvo = ef.max_sharpe()
        cleaned_weights_mvo = ef.clean_weights()
        print(dict(cleaned_weights_mvo))
        #ef.save_weights_to_file("weights.csv")  # saves to file
        performance = ef.portfolio_performance(verbose=True)
        output = {'raw weights': raw_weights_mvo, 'clean weights': cleaned_weights_mvo, "performance": performance}
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation MVO", exc_info=True)
def make_portfolio_hrp(df=pd.DataFrame):
    try:
        returns = df.pct_change().dropna()
        hrp = HRPOpt(returns)
        hrp_weights = hrp.optimize()
        performance=hrp.portfolio_performance(verbose=True)
        output = {'raw weights': hrp_weights, "performance": performance}
        print(dict(hrp_weights))
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation HRP", exc_info=True)
def make_portfolio_mcvar(df=pd.DataFrame):
    try:
        mu = mean_historical_return(df)
        S = portfolio.cov()
        ef_cvar = EfficientCVaR(mu, S)
        cvar_weights = ef_cvar.min_cvar()
        cleaned_weights = ef_cvar.clean_weights()
        print(dict(cleaned_weights))
        output = {'clean weights': cleaned_weights}
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation mCVAR", exc_info=True)

def make_portfolio_blacklitterman(df=pd.DataFrame):
    try:
        S = sample_cov(df)
        viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.131321}
        bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
        rets = bl.bl_returns()
        ef = EfficientFrontier(rets, S)
        ef.max_sharpe()
        weights = bl.clean_weights()
        # OR use return-implied weights
        print(dict(weights))
        output = {'weights': weights}
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation BL", exc_info=True)

def allocation_values(df,weights,total_portfolio):
    try:
        #calculate allocation values
        latest_prices = get_latest_prices(df)
        #Discrete allocation can deal for short positions
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio)
        allocation, leftover = da.greedy_portfolio()
        print("Discrete allocation:", allocation)
        remaining = ("Funds remaining: ${:.2f}".format(leftover))
        output = {'Discrete allocation': allocation, '.Remaining:': remaining}
        return output
    except Exception as e:
        log.error("Error in portfolio allocation", exc_info=True)

def get_stock_datareader(ticker):
    data = web.DataReader(ticker,"yahoo",START,END)
    data["ticker{}"] = data["Close"]
    data = data[[ticker]]
    print(data.head())
    return data

def get_stock_yahoo(ticker):
    data = yf.download(ticker, START, END)
    data[ticker] = data["Adj Close"]
    data = data[[ticker]]
    print(data.head())
    return data

def plot_price(df=pd.DataFrame):
    plt.figure(figsize=(14, 7))
    for c in df.columns.values:
        plt.plot(df.index, df[c], lw=3, alpha=0.8, label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in $')
    return plt.show()

def plot_return(df=pd.DataFrame):
    returns = df.pct_change()
    plt.figure(figsize=(14, 7))
    for c in returns.columns.values:
        plt.plot(returns.index, returns[c], lw=3, alpha=0.8, label=c)
    plt.legend(loc='upper right', fontsize=12)
    plt.ylabel('daily returns')
    return plt.show()

def plot_eff_frontier(df=pd.DataFrame):
    mu = mean_historical_return(df)
    S = portfolio.cov()
    ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
    ef.add_constraint(lambda w: w[0] >= 0.2)
    ef.add_constraint(lambda w: w[2] == 0.15)
    ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)
    # 100 portfolios with risks between 0.10 and 0.30. Range of parmeter for a frontier.
    risk_range = np.linspace(0.10, 0.40, 100)
    plotting.plot_efficient_frontier(ef, ef_param="risk", ef_param_range=risk_range,
                                     show_assets=True, showfig=True)
    fig, ax = plt.subplots()
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the tangency portfolio
    ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 10000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ef_scatter.png", dpi=200)
    return plt.show()

def combine_stocks(tickers):
    data_frames = []
    for i in tickers:
        data_frames.append(get_stock_yahoo(i))
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='outer'), data_frames)
    print(df_merged.head())
    return df_merged


def display(start,end,data):
    plt.figure(figsize=(20, 10))
    plt.title('Opening Prices from {} to {}'.format(start,end))
    plt.plot(data['Open'])
    return plt.show()

# PUT GLOBALS HERE
DF_PATH="https://raw.githubusercontent.com/artkochnev/project_capitalist/main/scripts/assets/stocks.csv?token=GHSAT0AAAAAABUHJOP2HUKK3JBWPBXDXI3WYT6BP6Q"
download = requests.get(DF_PATH).content
DF = pd.read_csv(io.StringIO(download.decode('utf-8')),parse_dates=True, index_col="date")
if DF.empty:
    DF_PATH = r'assets/stocks.csv'
    DF = pd.read_csv(DF_PATH, parse_dates=True, index_col="date")
    BUNDLE = Stocks(df = DF)



# FINAL SCRIPT
if __name__ == "__main__":
    #tickers = ["MRNA", "PFE", "JNJ"]
    #use the above DF for the below calculations
    #portfolio= combine_stocks(tickers)
    portfolio = combine_stocks(DF)
    plot_price=plot_price(portfolio)
    plot_return=plot_return(portfolio)
    plot_eff_frontier=plot_eff_frontier(portfolio)
    #Describe the 2 plots dinamically
    #MVO
    portfolio_weights_performance_mvo= make_portfolio_mvo(portfolio)
    weights = portfolio_weights_performance_mvo['clean weights']  # Use a selector by key inside the dictionary, does wonders.
    portfolio_allocation_mvo=allocation_values(portfolio,weights,10000)
    #HRP
    portfolio_weights_performance_hrp=make_portfolio_hrp(portfolio)
    weights = portfolio_weights_performance_mvo['raw weights']
    portfolio_allocation_hrp=allocation_values(portfolio,weights,10000)
    #mCVAR
    portfolio_weights_performance_mcvar=make_portfolio_mcvar(portfolio)
    weights = portfolio_weights_performance_mvo['clean weights']
    portfolio_allocation_mcvar=allocation_values(portfolio,weights,10000)