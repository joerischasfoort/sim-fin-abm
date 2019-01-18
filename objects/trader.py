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

    def sell(self, amount, price):
        """
        Sells `amount` of stocks for a total of `price`
        :param amount: int Number of stocks sold.
        :param price: float Total price for stocks.
        :return: -
        """
        if self.var.stocks[-1] < amount:
            raise ValueError("not enough stocks to sell this amount")
        self.var.stocks[-1] -= amount
        self.var.money[-1] += price

    def buy(self, amount, price):
        """
        Buys `amount` of stocks for a total of `price`
        :param amount: int number of stocks bought.
        :param price: float total price for stocks.
        :return: -
        """
        if self.var.money[-1] < price:
            raise ValueError("not enough money to buy this amount of stocks")

        self.var.stocks[-1] += amount
        self.var.money[-1] -= price


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
        self.active_orders = []


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


class TraderVariablesDistribution:
    """
    Holds the initial variables for the traders
    """
    def __init__(self, weight_fundamentalist, weight_chartist, weight_random, forecast_adjust,
                 money, stocks, covariance_matrix):
        """
        Initializes variables for the trader
        :param weight_fundamentalist: float fundamentalist expectation component
        :param weight_chartist: float trend-following chartism expectation component
        :param weight_random: float random or heterogeneous expectation component
        :param weight_mean_reversion: float mean-reversion chartism expectation component
        """
        self.weight_fundamentalist = weight_fundamentalist
        self.weight_chartist = weight_chartist
        self.weight_random = weight_random
        self.forecast_adjust = forecast_adjust
        self.money = [money] #TODO make into list to store history
        self.stocks = [stocks] #TODO make into list to store history
        self.covariance_matrix = covariance_matrix
        self.active_orders = []


class TraderParametersDistribution:
    """
    Holds the the trader parameters for the distribution model
    """

    def __init__(self, ref_horizon, risk_aversion, max_spread):
        """
        Initializes trader parameters
        :param ref_horizon: integer horizon over which the trader can observe the past
        :param max_spread: Maximum spread at which the trader will submit orders to the book
        :param risk_aversion: float aversion to price volatility
        """
        self.horizon = ref_horizon
        self.risk_aversion = risk_aversion
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
        self.returns = {'stocks': 0.0, 'money': 0.0}

