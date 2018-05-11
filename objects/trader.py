import numpy as np


class Trader:
    """Class holding low frequency trader properties"""
    def __init__(self, name, variables, parameters, expectations):
        self.name = name
        self.var = variables
        self.par = parameters
        self.exp = expectations

    def __repr__(self):
        return 'Trader' + str(self.name)


class Tradervariables:
    """
    Holds the initial variables for the traders
    """
    def __init__(self, weight_fundamentalist, weight_chartist, weight_random, weight_mean_reversion, weight_buy_hold):
        self.weight_fundamentalist = abs(np.random.laplace(0., weight_fundamentalist))
        self.weight_chartist = abs(np.random.laplace(0., weight_chartist))
        self.weight_random = abs(np.random.laplace(0., weight_random))
        self.weight_mean_reversion = abs(np.random.laplace(0., weight_mean_reversion))
        self.weight_buy_hold = abs(np.random.laplace(0., weight_buy_hold))
        self.forecast_adjust = 1. / (
            self.weight_fundamentalist + self.weight_chartist + self.weight_random + self.weight_mean_reversion + self.weight_buy_hold)
        self.last_buy_price = {'price': 0, 'age': 0}


class TraderParameters:
    """
    Holds the the trader parameters
    """
    def __init__(self, horizon_min, horizon_max, max_spread):
        self.horizon = np.random.randint(horizon_min, horizon_max)
        self.spread = max_spread * np.random.rand()


class TraderExpectations:
    """
    Holds the agent expectations for several variables
    """
    def __init__(self, price):
        self.price = price

