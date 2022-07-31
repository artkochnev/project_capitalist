import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import CLA, plotting

import pandas as pd

LINK = "https://raw.githubusercontent.com/robertmartin8/PyPortfolioOpt/master/tests/resources/stock_prices.csv"

df = pd.read_csv(LINK, index_col="date")

x = df.loc[df.index < '2014-09-24']
print(x)
