# THE PORTFOLIO SELECTION ALGRORITHM
import pandas as pd
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
"""CHECK SCRIPTS IN THIS LIBRARY
https://pypi.org/project/pyportfolioopt/#an-overview-of-classical-portfolio-optimization-methods
https://builtin.com/data-science/portfolio-optimization-python
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
def make_portfolio(df = pd.DataFrame):
    try:
        # Calculate expected returns and sample covariance
        mu = mean_historical_return(df)
        S = sample_cov(df)
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        print(dict(cleaned_weights))
        #ef.save_weights_to_file("weights.csv")  # saves to file
        performance = ef.portfolio_performance(verbose=True)
        output = {'raw weights': raw_weights, 'clean weights': cleaned_weights, "performance": performance}
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation", exc_info=True)

def allocation_values(df,weights,total_portfolio):
    try:
        #calculate allocation values
        latest_prices = get_latest_prices(df)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio)
        allocation, leftover = da.greedy_portfolio()
        print("Discrete allocation:", allocation)
        remaining = ("Funds remaining: ${:.2f}".format(leftover))
        output = {'Discrete allocation': allocation}
        return output
    except Exception as e:
        log.error("Error in portfolio allocation", exc_info=True)

def get_stock_datareader(ticker):
    data = web.DataReader("ticker{}","yahoo",START,END)
    data["ticker{}"] = data["Close"]
    data = data[["ticker{}"]]
    print(data.head())
    return data

def get_stock_yahoo(ticker):
    data = yf.download(ticker, START, END)
    data[ticker] = data["Adj Close"]
    data = data[[ticker]]
    print(data.head())
    return data

def plot_price(df):
    plt.figure(figsize=(14, 7))
    for c in df.columns.values:
        plt.plot(df.index, df[c], lw=3, alpha=0.8, label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in $')

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
    plt.show()

# PUT GLOBALS HERE
DF_PATH = r'assets/stocks.csv'
#DF = pd.read_csv(DF_PATH, parse_dates=True, index_col="date")
#BUNDLE = Stocks(df = DF)

# FINAL SCRIPT
if __name__ == "__main__":
    tickers = ["MRNA", "PFE", "JNJ"]
    portfolio= combine_stocks(tickers)
    #plot_price=plot_price(portfolio)
    #plot_return
    #https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
    portfolio_weights_performance= make_portfolio(portfolio)
    weights = portfolio_weights_performance['clean weights']  # Use a selector by key inside the dictionary, does wonders.
    portfolio_allocation=allocation_values(portfolio,weights,10000)
    s=1
    #portfolio_info = make_portfolio(DF)
    #print(portfolio_info['clean weights'])
    #print(portfolio_info['performance'])

