from objects.trader import *
from objects.orderbook import *


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
