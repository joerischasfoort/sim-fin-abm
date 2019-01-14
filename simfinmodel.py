""""The main model"""
import numpy as np
import random


np.seterr(all='raise')

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
        chartist_component = np.cumsum(orderbook.returns[:-parameters['horizon_max']-1:-1]
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


def sim_fin_model_optimized(traders, parameters, seed=1):
    """
    The main model function optimized using numpy
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    np.random.seed(seed)
    fundamental = np.zeros(parameters["ticks"])
    fundamental[0:parameters['horizon_max']] = parameters["fundamental_value"]

    close_prices = np.zeros(parameters["ticks"])
    close_prices[0:parameters['horizon_max']] = parameters["fundamental_value"]
    returns = np.zeros(parameters["ticks"])

    bids_prices = np.zeros(parameters['trader_sample_size'])
    bids_volumes = np.zeros(parameters['trader_sample_size'])
    bids_ages = np.zeros(parameters['trader_sample_size'])
    asks_prices = np.zeros(parameters['trader_sample_size'])
    asks_volumes = np.zeros(parameters['trader_sample_size'])
    asks_ages = np.zeros(parameters['trader_sample_size'])

    # match highest bid with lowest ask
    def match_orders():
        # First, make sure that neither the bids or asks books are empty
        if bids_prices.sum() <= 0 or asks_prices.sum() <= 0:
            return bids_prices, bids_volumes, bids_ages, asks_prices, asks_volumes, asks_ages

        # locate
        best_bid = bids_prices.argmax()
        best_ask = asks_prices[(asks_prices > 0)].argmin()

        while bids_prices[best_bid] >= asks_prices[best_ask]:
            # price is winning ask price and volume is lowest volume
            transaction_price = asks_prices[best_ask]
            transaction_volume = np.array([asks_volumes[best_ask], bids_volumes[best_bid]]).min()

            # subtract volume from orders
            bids_volumes[best_bid] = bids_volumes[best_bid] - transaction_volume
            asks_volumes[best_ask] = asks_volumes[best_ask] - transaction_volume

            # if volume depleted delete order from, price, age, and volume if vol == 0
            if bids_volumes[best_bid] <= 0:
                bids_prices[best_bid] = 0
                bids_ages[best_bid] = 0

            if asks_volumes[best_ask] <= 0:
                asks_prices[best_ask] = np.inf
                asks_ages[best_ask] = 0

            # save price and volume
            transaction_prices.append(transaction_price)
            transaction_volumes.append(transaction_volume)

            # determine new best bid and best ask
            best_bid = bids_prices.argmax()
            best_ask = asks_prices[(asks_prices > 0)].argmin()

        asks_prices[(asks_prices == np.inf)] = 0

        # return leftover bids
        bids_left = (bids_volumes != 0)
        a_bids = len(bids_left[bids_left == True])
        asks_left = (asks_volumes != 0)
        a_asks = len(asks_left[asks_left == True])

        l_bids_p = np.zeros(parameters['trader_sample_size'] + a_bids)
        l_bids_p[:a_bids] = bids_prices[bids_left]
        l_bids_v = np.zeros(parameters['trader_sample_size'] + a_bids)
        l_bids_v[:a_bids] = bids_volumes[bids_left]
        l_bids_a = np.zeros(parameters['trader_sample_size'] + a_bids)
        l_bids_a[:a_bids] = bids_ages[bids_left] + 1  # age orders by 1

        l_asks_p = np.zeros(parameters['trader_sample_size'] + a_asks)
        l_asks_p[:a_asks] = asks_prices[asks_left]
        l_asks_v = np.zeros(parameters['trader_sample_size'] + a_asks)
        l_asks_v[:a_asks] = asks_volumes[asks_left]
        l_asks_a = np.zeros(parameters['trader_sample_size'] + a_asks)
        l_asks_a[:a_asks] = asks_ages[asks_left] + 1  # age orders by 1

        # clear orders that are too old
        n_bids_a = l_bids_a[(l_bids_a < parameters['max_order_expiration_ticks'])]
        n_bids_p = l_bids_p[(l_bids_a < parameters['max_order_expiration_ticks'])]
        n_bids_v = l_bids_v[(l_bids_a < parameters['max_order_expiration_ticks'])]
        n_asks_a = l_asks_a[(l_asks_a < parameters['max_order_expiration_ticks'])]
        n_asks_p = l_asks_p[(l_asks_a < parameters['max_order_expiration_ticks'])]
        n_asks_v = l_asks_v[(l_asks_a < parameters['max_order_expiration_ticks'])]

        return n_bids_p, n_bids_v, n_bids_a, n_asks_p, n_asks_v, n_asks_a

    for tick in range(parameters['horizon_max'], parameters["ticks"]):
        # update the fundamental value via random walk process
        fundamental[tick] = fundamental[tick - 1] + parameters["std_fundamental"] * np.random.randn()
        # select random sample of active traders
        a = np.arange(0, len(traders), parameters['trader_sample_size'])
        np.random.shuffle(a)
        active_traders = traders[a[:parameters['trader_sample_size']]]

        # update common expectation components
        fundamental_component = np.log(fundamental[tick] / close_prices[tick-1])

        # create an array of chartist component for every trader based individual horizons and noise components
        individual_chartist_components = np.cumsum(np.flip(returns, 0))[active_traders.horizon] / (active_traders.horizon + 1)
        individual_noise_components = np.random.normal(0.0, parameters["std_noise"], parameters["trader_sample_size"])

        # update all fcast returns
        fcast_returns = active_traders.forecast_adjust * (active_traders.weight_fundamentalist * fundamental_component +
                                                          active_traders.weight_chartist * individual_chartist_components +
                                                          active_traders.weight_random * individual_noise_components -
                                                          active_traders.weight_mean_reversion * individual_chartist_components
                                                          )
        fcast_prices = close_prices[tick-1] * np.exp(fcast_returns)

        # create masks to determine bidding and asking agents
        bidding_agents = (fcast_prices > close_prices[tick-1])
        amount_bids = len(bidding_agents[bidding_agents == True])
        asking_agents = (fcast_prices < close_prices[tick-1])
        amount_asks = len(asking_agents[asking_agents == True])

        # find position of first-non zero
        def non_zero(arr):
            if arr.sum() > 0:
                return np.nonzero(arr)[0][0] + 1
            else:
                return 0

        # enter bids right of the existing bids
        bids_prices[non_zero(bids_prices):non_zero(bids_prices) + amount_bids] = fcast_prices[bidding_agents] * (1. - active_traders.spread[bidding_agents])
        bids_volumes[non_zero(bids_volumes):non_zero(bids_volumes) + amount_bids] = np.random.randint(1, parameters['std_vol'], amount_bids)
        bids_ages[non_zero(bids_ages):non_zero(bids_ages) + amount_bids] = np.ones(amount_bids)
        # enter asks right of existing asks
        asks_prices[non_zero(asks_prices):non_zero(asks_prices) + amount_asks] = fcast_prices[asking_agents] * (1. + active_traders.spread[asking_agents])
        asks_volumes[non_zero(asks_volumes):non_zero(asks_volumes) + amount_asks] = np.random.randint(1, parameters['std_vol'], amount_asks)
        asks_ages[non_zero(asks_ages):non_zero(asks_ages) + amount_asks] = np.ones(amount_asks)

        # save daily prices and volumes
        transaction_prices = []
        transaction_volumes = []

        # call match orders function
        bids_prices, bids_volumes, bids_ages, asks_prices, asks_volumes, asks_ages = match_orders()

        # store relevant data
        if transaction_prices:
            close_prices[tick] = transaction_prices[-1]
            returns[tick] = close_prices[tick-1] - close_prices[tick-2]
        else:
            close_prices[tick] = close_prices[tick-1]
            returns[tick] = returns[tick-1]

    return traders, close_prices, returns


