import random
import numpy as np
import scipy.optimize
from functions.portfolio_optimization import *
from functions.helpers import calculate_covariance_matrix, div0


def pb_distr_model(traders, orderbook, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    # TODO the next can be done in a more elegant way
    orderbook.tick_close_price.append(fundamental[-1])

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1): # for init history
        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1]) # TODO debug
            trader.var.weight_fundamentalist.append(trader.var.weight_fundamentalist[-1])
            trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            trader.var.weight_random.append(trader.var.weight_random[-1])

        # sort the traders by wealth to
        traders.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price]) #TODO debug
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2] #TODO debug
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                def evolve(probability):
                    return random.random() < probability

                # Evolve an expectations parameter by learning from a successful trader
                if evolve(trader.par.learning_ability):
                    wealthy_trader = traders[random.randint(0, parameters['trader_sample_size'])]
                    # update weights
                    trader.var.weight_fundamentalist[-1] = np.mean([wealthy_trader.var.weight_fundamentalist[-1] * wealthy_trader.var.forecast_adjust,
                                                           trader.var.weight_fundamentalist[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust
                    trader.var.weight_chartist[-1] = np.mean([wealthy_trader.var.weight_chartist[-1] * wealthy_trader.var.forecast_adjust,
                                                     trader.var.weight_chartist[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust
                    trader.var.weight_random[-1] = np.mean([wealthy_trader.var.weight_random[-1] * wealthy_trader.var.forecast_adjust,
                                                   trader.var.weight_random[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust

                # mutate an expectations parameter
                if evolve(parameters['mutation_probability']):
                    expectation_components = [trader.var.weight_fundamentalist, trader.var.weight_chartist, trader.var.weight_random]
                    mutation_parameters = [parameters['w_fundamentalists'], parameters['w_momentum'], parameters['w_random']]
                    index_mutation = random.randint(0, len(expectation_components)-1)
                    expectation_components[index_mutation][-1] = abs(np.random.laplace(expectation_components[index_mutation][-1],#mutation_parameters[index_mutation], TODO check if works as intended
                                                                               mutation_parameters[index_mutation] ** 2))
                    # recalculate the traders forecast adjustment
                    trader.var.forecast_adjust = 1. / (trader.var.weight_fundamentalist[-1] + trader.var.weight_chartist[-1] + trader.var.weight_random[-1])

                # record sentiment in orderbook
                orderbook.sentiment.append(trader.var.forecast_adjust * np.array([trader.var.weight_fundamentalist[-1],
                                                                                  trader.var.weight_chartist[-1],
                                                                                  trader.var.weight_random[-1]]))

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation #TODO DEBUG
                trader.exp.returns['stocks'] = trader.var.forecast_adjust * (
                    trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                    trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                    trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook


def pb_distr_model_shock(traders, orderbook, parameters, shock, shock_period, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    # TODO the next can be done in a more elegant way
    orderbook.tick_close_price.append(fundamental[-1])

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1):# for init history
        if tick == shock_period:
            print('Apply shock in simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1]) #TODO debug
            trader.var.weight_fundamentalist.append(trader.var.weight_fundamentalist[-1])
            if tick == shock_period + parameters['horizon']:
                # turn half of the population into hardcore chartists
                if random.random() < 0.5:
                    trader.var.weight_chartist.append(abs(np.random.laplace(shock, shock)))
                else:
                    trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            else:
                trader.var.weight_chartist.append(trader.var.weight_chartist[-1])
            trader.var.weight_random.append(trader.var.weight_random[-1])

        # sort the traders by wealth to
        traders.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                def evolve(probability):
                    return random.random() < probability

                # Evolve an expectations parameter by learning from a successful trader
                if evolve(trader.par.learning_ability):
                    wealthy_trader = traders[random.randint(0, parameters['trader_sample_size'])]
                    # update weights
                    trader.var.weight_fundamentalist[-1] = np.mean([wealthy_trader.var.weight_fundamentalist[-1] * wealthy_trader.var.forecast_adjust,
                                                           trader.var.weight_fundamentalist[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust
                    trader.var.weight_chartist[-1] = np.mean([wealthy_trader.var.weight_chartist[-1] * wealthy_trader.var.forecast_adjust,
                                                     trader.var.weight_chartist[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust
                    trader.var.weight_random[-1] = np.mean([wealthy_trader.var.weight_random[-1] * wealthy_trader.var.forecast_adjust,
                                                   trader.var.weight_random[-1] * trader.var.forecast_adjust]) / trader.var.forecast_adjust

                # mutate an expectations parameter
                if evolve(parameters['mutation_probability']):
                    expectation_components = [trader.var.weight_fundamentalist, trader.var.weight_chartist, trader.var.weight_random]
                    mutation_parameters = [parameters['w_fundamentalists'], parameters['w_momentum'], parameters['w_random']]
                    index_mutation = random.randint(0, len(expectation_components)-1)
                    expectation_components[index_mutation][-1] = abs(np.random.laplace(expectation_components[index_mutation][-1],#mutation_parameters[index_mutation], TODO check if works as intended
                                                                               mutation_parameters[index_mutation] ** 2))
                    # recalculate the traders forecast adjustment
                    trader.var.forecast_adjust = 1. / (trader.var.weight_fundamentalist[-1] + trader.var.weight_chartist[-1] + trader.var.weight_random[-1])

                # record sentiment in orderbook
                orderbook.sentiment.append(trader.var.forecast_adjust * np.array([trader.var.weight_fundamentalist[-1],
                                                                                  trader.var.weight_chartist[-1],
                                                                                  trader.var.weight_random[-1]]))

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = trader.var.forecast_adjust * (
                    trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                    trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                    trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook