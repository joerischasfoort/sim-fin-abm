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


def sim_fin_model_optimized(traders, parameters, seed=1):
    """
    The main model function optimized using numpy
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = np.zeros(parameters["fundamental_value"])

    mid_price = parameters["fundamental_value"]
    returns = #TODO initialize returns

    max_orderbook_lenght = parameters['max_order_expiration_ticks'] * parameters['trader_sample_size']
    bids_prices = np.zeros(max_orderbook_lenght)
    bids_volumes = np.zeros(max_orderbook_lenght)
    asks_prices = np.zeros(max_orderbook_lenght)
    asks_volumes = np.zeros(max_orderbook_lenght)

    for tick in range(parameters['horizon_max'] + 1, parameters["ticks"]):
        # update the fundamental value via random walk process
        fundamental[tick] = fundamental[tick - 1] + parameters["std_fundamental"] * np.random.randn()
        # select random sample of active traders
        active_traders = traders[[np.random.choice(np.arange(0, len(traders)), parameters['trader_sample_size'])]]

        # update common expectation components
        #mid_price = orderbook.tick_close_price[-1]
        fundamental_component = np.log(fundamental[-1] / mid_price)
        chartist_component = np.cumsum(returns[-parameters['horizon_max']:]
                                       ) / np.arange(1., float(parameters['horizon_max'] + 1))
        # create an array of chartist component for every trader based individual horizons
        individual_chartist_components = # TODO
        # create array of unique noise components per trader
        individual_noise_components = np.random.normal(0.0, parameters["std_noise"], parameters["trader_sample_size"])

        # update all fcast returns
        fcast_returns = active_traders.forecast_adjust * (active_traders.weight_fundamentalist * fundamental_component +
                                                          active_traders.weight_chartist * individual_chartist_components +
                                                          active_traders.weight_random * individual_noise_components -
                                                          active_traders.weight_mean_reversion * individual_chartist_components
                                                          )
        fcast_prices = mid_price * np.exp(fcast_returns)

        # create masks to determine bidding and asking agents
        bidding_agents = (fcast_prices > mid_price)
        asking_agents = (fcast_prices < mid_price)

        # enter bids right of the existing bids
        bids_prices[np.nonzero(bids_prices)[0]:np.nonzero(bids_prices)[0] + parameters['trader_sample_size']] = fcast_prices * (1. - active_traders.spread)
        bids_volumes[np.nonzero(bids_prices)[0]:np.nonzero(bids_prices)[0] + parameters['trader_sample_size']] = np.random.randint(0, parameters['std_vol'], parameters['trader_sample_size'])
        bids_ages = np.zeros(parameters['trader_sample_size'])
        # enter asks right of existing asks
        asks_prices[np.nonzero(bids_prices)[0]:np.nonzero(bids_prices)[0] + parameters['trader_sample_size']] = fcast_prices * (1. + active_traders.spread)
        asks_volumes[np.nonzero(bids_prices)[0]:np.nonzero(bids_prices)[0] + parameters['trader_sample_size']] = np.random.randint(0, parameters['std_vol'], parameters['trader_sample_size'])
        asks_ages[np.nonzero(bids_prices)[0]:np.nonzero(bids_prices)[0] + parameters['trader_sample_size']] = np.zeros(parameters['trader_sample_size'])

        # save daily prices and volumes
        transaction_prices = []
        transaction_volumes = []
        # match highest bid with lowest ask
        def match_orders():
            # locate
            best_bid = bids_prices.argmax()
            best_ask = asks_prices.argmin()

            if best_bid >= best_ask:
                # price is winning ask price
                transaction_price = asks_prices[best_ask]

                # subtract volume of lowest vol
                transaction_volume = np.min(np.array(asks_volumes[best_ask], bids_volumes[best_bid]))

                # save price and volume
                transaction_prices.append(transaction_price)
                transaction_volumes.append(transaction_volume)


        # age all orders
        bids_ages = bids_ages + 1
        asks_ages = asks_ages + 1
        # clear expired orders
        # aka only keep existing orders in new arrays
        bids_prices =
        bids_volumes =
        bids_ages =
        asks_prices =
        asks_volumes =
        asks_ages =
        # store relevant data
        data =

    return traders, data