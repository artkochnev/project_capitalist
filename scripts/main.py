# THE PORTFOLIO SELECTION ALGRORITHM
import pandas as pd
import logging
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

"""CHECK SCRIPTS IN THIS LIBRARY
https://pypi.org/project/pyportfolioopt/#an-overview-of-classical-portfolio-optimization-methods
https://builtin.com/data-science/portfolio-optimization-python
"""

# LOGGER
log = logging.getLogger(__name__)
log.debug

# PUT NEW CLASSES HERE
class Stocks:
    def __init__(self, df = pd.DataFrame):
        self.df = df

# PUT NEW FUNCTIONS HERE
def make_portfolio(df = pd.DataFrame):
    try:
        # Calculate expected returns and sample covariance
        mu = expected_returns.mean_historical_return(df)
        S = risk_models.sample_cov(df)

        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        #ef.save_weights_to_file("weights.csv")  # saves to file
        performance = ef.portfolio_performance(verbose=True)
        output = {'raw weights': raw_weights, 'clean weights': cleaned_weights, "performance": performance}
        return(output)
    except Exception as e:
        log.error("Error in portfolio calculation", exc_info=True)

# PUT GLOBALS HERE
DF_PATH = r'assets/stocks.csv'
DF = pd.read_csv(DF_PATH, parse_dates=True, index_col="date")
BUNDLE = Stocks(df = DF)

# FINAL SCRIPT
if __name__ == "__main__":
    portfolio_info = make_portfolio(DF)
    print(portfolio_info['clean weights'])
    print(portfolio_info['performance'])
