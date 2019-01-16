from objects.trader import *
from objects.orderbook import *
import pandas as pd
import numpy as np
from functions.helpers import calculate_covariance_matrix



def init_objects(parameters, seed):
    """
    Initialises the model agents and orderbook
    :param parameters: object which holds all model parameters
    :param seed: integer seed for the random number generator
    :return: list of agents
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    for idx in range(n_traders):
        weight_fundamentalist = parameters['w_fundamentalists']
        weight_chartist = parameters['w_momentum']
        weight_random = parameters['w_random']
        weight_mean_reversion = parameters['w_mean_reversion']
        lft_vars = Tradervariables(weight_fundamentalist, weight_chartist, weight_random,
                                   weight_mean_reversion)
        lft_params = TraderParameters(1, parameters['horizon_max'], parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["spread_max"],
                               parameters['horizon_max'], parameters['max_order_expiration_ticks'])

    return traders, orderbook


def init_objects_contrarians(parameters, seed):
    """
    Initialises the model agents and orderbook for experiment which replaces fundamentalists with mean
    reversion traders
    :param parameters: object which holds all model parameters
    :param seed: integer seed for the random number generator
    :return: list of agents, Orderbook object
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    weight_fundamentalist = (1 - parameters['share_mr']) * parameters['weight_contrarians']
    weight_mean_reversion = parameters['share_mr'] * parameters['weight_contrarians']

    for idx in range(n_traders):
        weight_chartist = parameters['w_momentum']
        weight_random = parameters['w_random']
        lft_vars = Tradervariables(weight_fundamentalist, weight_chartist, weight_random,
                                   weight_mean_reversion)
        lft_params = TraderParameters(1, parameters['horizon_max'], parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["spread_max"],
                               parameters['horizon_max'], parameters['max_order_expiration_ticks'])

    return traders, orderbook


def init_objects_chartists(parameters, seed):
    """
    Initialises the model agents and orderbook for experiment which replaces fundamentalists with mean
    reversion traders
    :param parameters: object which holds all model parameters
    :param seed: integer seed for the random number generator
    :return: list of agents, Orderbook object
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    weight_chartist = (1 - parameters['share_mr']) * parameters['weight_chartists']
    weight_mean_reversion = parameters['share_mr'] * parameters['weight_chartists']

    for idx in range(n_traders):
        weight_fundamentalist = parameters['w_fundamentalists']
        weight_random = parameters['w_random']
        lft_vars = Tradervariables(weight_fundamentalist, weight_chartist, weight_random,
                                   weight_mean_reversion)
        lft_params = TraderParameters(1, parameters['horizon_max'], parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["spread_max"],
                               parameters['horizon_max'], parameters['max_order_expiration_ticks'])

    return traders, orderbook


def init_objects_optimized(parameters, seed):
    """
    Initialises the model agents and orderbook using numpy dtypes, this will greatly enhance simulation performance
    :param parameters: object which holds all model parameters
    :param seed: integer seed for the random number generator
    :return: list of agents
    """
    np.random.seed(seed)

    agent_def = [('name', 'S6'), ('weight_fundamentalist', 'f8'),
                 ('weight_chartist', 'f8'), ('weight_random', 'f8'),
                 ('weight_mean_reversion', 'f8'), ('forecast_adjust', 'f8'), ('horizon', 'i8'),
                 ('spread', 'f8'), ('exp_price', 'f8')]

    init_traders = []
    for i in range(parameters["n_traders"]):
        name = 'ag{}'.format(i)
        weight_fundamentalist = abs(np.random.normal(0., parameters["w_fundamentalists"]))
        weight_chartist = abs(np.random.normal(0., parameters["w_momentum"]))
        weight_random = abs(np.random.normal(0., parameters["w_random"]))
        weight_mean_reversion = abs(np.random.normal(0., parameters["w_mean_reversion"]))
        f_cast_adj = 1. / (weight_fundamentalist + weight_chartist + weight_random + weight_mean_reversion)
        horizon = np.random.randint(1, parameters['horizon_max'])
        spread = parameters['spread_max'] * np.random.rand()
        exp_price = parameters['fundamental_value']

        init_traders.append((name, weight_fundamentalist, weight_chartist, weight_random, weight_mean_reversion,
                             f_cast_adj, horizon, spread, exp_price))

    traders = np.rec.array(init_traders, dtype=agent_def)

    return traders


def init_objects_distr(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    max_horizon = parameters['horizon'] * 2  # this is the max horizon of an agent if 100% fundamentalist
    historical_stock_returns = np.random.normal(0, parameters["std_fundamental"], max_horizon)

    for idx in range(n_traders):
        weight_fundamentalist = abs(np.random.laplace(parameters['w_fundamentalists'], parameters['w_fundamentalists']**2))
        weight_chartist = np.random.laplace(parameters['w_momentum'], parameters['w_momentum']**2)
        weight_random = np.random.laplace(parameters['w_random'], parameters['w_random']**2)
        forecast_adjust = 1. / (weight_fundamentalist + weight_chartist + weight_random)

        init_stocks = int(np.random.uniform(0, parameters["init_stocks"]))
        init_money = np.random.uniform(0, (parameters["init_stocks"] * parameters['fundamental_value']))

        # initialize co_variance_matrix
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["std_fundamental"])

        lft_vars = TraderVariablesDistribution(weight_fundamentalist, weight_chartist, weight_random, forecast_adjust,
                                               init_money, init_stocks, init_covariance_matrix)

        # determine heterogeneous horizon and risk aversion based on
        relative_fundamentalism = np.divide(1 + (abs(weight_fundamentalist * forecast_adjust)),
                                            1 + (abs(weight_chartist * forecast_adjust)))
        individual_horizon = min(int(parameters['horizon'] * relative_fundamentalism), max_horizon)
        individual_risk_aversion = parameters["base_risk_aversion"] * relative_fundamentalism

        lft_params = TraderParametersDistribution(individual_horizon, individual_risk_aversion, parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["std_fundamental"],
                               max_horizon,
                               parameters['ticks'])

    # initialize order-book returns for initial variance calculations
    orderbook.returns = list(historical_stock_returns)

    return traders, orderbook