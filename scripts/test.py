import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import CLA, plotting

import pandas as pd

LINK = "https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/master/tests/resources/stock_prices.csv"

df = pd.read_csv(LINK, index_col="date")
mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()

cla = CLA(mu, S)
cla.max_sharpe()
cla.portfolio_performance(verbose=True)
ax = plotting.plot_efficient_frontier(cla, showfig=True)

print_list = [df, mu, S]

for p in print_list:
    print(p)


