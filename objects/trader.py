import numpy as np


class Trader:
    """Class holding low frequency trader properties"""
    def __init__(self, name, variables, parameters, expectations):
        """
        Initialize trader class
        :param name: integer number which will be the name of the trader
        :param variables: object Tradervariables
        :param parameters: object TraderParameters
        :param expectations: object TraderExpectations
        """
        self.name = name
        self.var = variables
        self.par = parameters
        self.exp = expectations

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return 'Trader' + str(self.name)


class Tradervariables:
    """
    Holds the initial variables for the traders
    """
    def __init__(self, weight_fundamentalist, weight_chartist, weight_random, weight_mean_reversion):
        """
        Initializes variables for the trader
        :param weight_fundamentalist: float fundamentalist expectation component
        :param weight_chartist: float trend-following chartism expectation component
        :param weight_random: float random or heterogeneous expectation component
        :param weight_mean_reversion: float mean-reversion chartism expectation component
        """
        self.weight_fundamentalist = abs(np.random.laplace(0., weight_fundamentalist))
        self.weight_chartist = abs(np.random.laplace(0., weight_chartist))
        self.weight_random = abs(np.random.laplace(0., weight_random))
        self.weight_mean_reversion = abs(np.random.laplace(0., weight_mean_reversion))
        self.forecast_adjust = 1. / (
            self.weight_fundamentalist + self.weight_chartist + self.weight_random + self.weight_mean_reversion)
        self.last_buy_price = {'price': 0, 'age': 0}


class TraderParameters:
    """
    Holds the the trader parameters
    """
    def __init__(self, horizon_min, horizon_max, max_spread):
        """
        Initializes trader parameters
        :param horizon_min: integer minimum horizon over which the trader can observe the past
        :param horizon_max: integer maximum horizon over which the trader can observe the past
        :param max_spread: Maximum spread at which the trader will submit orders to the book
        """
        self.horizon = np.random.randint(horizon_min, horizon_max)
        self.spread = max_spread * np.random.rand()


class TraderExpectations:
    """
    Holds the agent expectations for several variables
    """
    def __init__(self, price):
        """
        Initializes trader expectations
        :param price: float
        """
        self.price = price

