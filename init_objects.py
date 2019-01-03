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

    for idx in range(n_traders):
        weight_fundamentalist = parameters['w_fundamentalists']
        weight_chartist = parameters['w_momentum']
        weight_random = parameters['w_random']
        lft_vars = TraderVariablesDistribution(weight_fundamentalist, weight_chartist, weight_random,
                                               parameters["init_money"], parameters["init_stocks"])
        individual_horizon = int(parameters['horizon'] * np.divide(1 + weight_fundamentalist, 1 + weight_chartist)) # equation 4 TODO Debug NEW
        lft_params = TraderParametersDistribution(individual_horizon, parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["spread_max"],
                               parameters['horizon'] + parameters["horizon"], #TODO this is not an elegant solution
                               parameters['max_order_expiration_ticks'])

    return traders, orderbook