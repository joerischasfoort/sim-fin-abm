""""The main model"""
import numpy as np
import random


def sim_fin_model(traders, orderbook, parameters, seed=1):
    """
    The main model function
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]

    for tick in range(parameters['horizon_max'] + 1, parameters["ticks"]):
        # evolve the fundamental value via random walk process
        fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())

        # select random sample of active traders
        active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

        # update common expectation components
        mid_price = orderbook.tick_close_price[-1]
        fundamental_component = np.log(fundamental[-1] / mid_price)
        chartist_component = np.cumsum(orderbook.returns[-parameters['horizon_max']:]
                                       ) / np.arange(1., float(parameters['horizon_max'] + 1))

        for trader in active_traders:
            # update trader specific expectations
            noise_component = parameters['std_noise'] * np.random.randn()

            fcast_return = trader.var.forecast_adjust * (
                trader.var.weight_fundamentalist * fundamental_component +
                trader.var.weight_chartist * chartist_component[trader.par.horizon] +
                trader.var.weight_random * noise_component -
                trader.var.weight_mean_reversion * chartist_component[trader.par.horizon])

            fcast_price = mid_price * np.exp(fcast_return)

            # submit orders
            if fcast_price > mid_price:
                bid_price = fcast_price * (1. - trader.par.spread)
                orderbook.add_bid(bid_price, abs(int(np.random.normal(scale=parameters['std_vol']))), trader)
            elif fcast_price < mid_price:
                ask_price = fcast_price * (1 + trader.par.spread)
                orderbook.add_ask(ask_price, abs(int(np.random.normal(scale=parameters['std_vol']))), trader)

        # match orders in the order-book
        while True:
            matched_orders = orderbook.match_orders()
            if matched_orders is None:
                break

        # clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook


