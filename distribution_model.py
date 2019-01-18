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

        # evolve the fundamental value via random walk process
        fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            # update common expectation components


            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price]) #TODO debug
            fundamental_component = np.log(
                fundamental[-1] / mid_price)  # TODO add expected mean reversion time period? as in equation 1

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2] #TODO debug
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))

            for trader in active_traders:
                # 1 Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = trader.var.forecast_adjust * (
                    trader.var.weight_fundamentalist * fundamental_component +
                    trader.var.weight_chartist * chartist_component[trader.par.horizon - 1] +
                    trader.var.weight_random * noise_component)
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

#
# def distr_model(traders, orderbook, parameters, seed=1):
#     """
#     The main model function of distribution model where trader stocks are tracked.
#     :param traders: list of Agent objects
#     :param orderbook: object Order book
#     :param parameters: dictionary of parameters
#     :param seed: integer seed to initialise the random number generators
#     :return: list of simulated Agent objects, object simulated Order book
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     fundamental = [parameters["fundamental_value"]]
#
#     for tick in range(parameters['horizon'] + 1, parameters["ticks"]):
#         if tick == 500:
#             print(500)
#
#         # evolve the fundamental value via random walk process
#         fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())
#
#         # select random sample of active traders
#         active_traders = random.sample(traders, int((parameters['trader_sample_size'])))
#
#         # update common expectation components
#         mid_price = orderbook.tick_close_price[-1]
#         fundamental_component = np.log(fundamental[-1] / mid_price) #TODO add expected mean reversion time period? as in equation 1
#         chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns)-1:-1]
#                                        ) / np.arange(1., float(len(orderbook.returns) + 1))
#
#         for trader in active_traders:
#             # 1 cancel any active orders
#             if trader.var.active_orders:
#                 for order in trader.var.active_orders:
#                     orderbook.cancel_order(order)
#                 trader.var.active_orders = []
#
#             # update trader specific expectations
#             noise_component = parameters['std_noise'] * np.random.randn()
#
#             fcast_return = trader.var.forecast_adjust * (
#                 trader.var.weight_fundamentalist * fundamental_component +
#                 trader.var.weight_chartist * chartist_component[trader.par.horizon - 1] +
#                 trader.var.weight_random * noise_component)
#
#             fcast_price = mid_price * np.exp(fcast_return)
#             fcast_volatility = np.var(orderbook.returns[-trader.par.horizon:])
#
#             def optimal_p_star(price):
#                 """Determine the price at which the hfm would be satisfied with its current portfolio (eq 10)"""
#                 stocks = np.divide(np.log(fcast_price / price),
#                                    trader.par.risk_aversion * fcast_volatility * price) - trader.var.stocks
#                 return stocks
#
#             def minimal_p(price):
#                 stocks_minus_cash = np.divide(np.log(fcast_price / price),
#                                               trader.par.risk_aversion * fcast_volatility * price
#                                               ) - trader.var.stocks - trader.var.money
#                 return stocks_minus_cash
#
#             def optimal_stock_holdings(price):
#                 """Determine the number of stocks a trader wants to hold at a given price (eq 8)"""
#                 stocks = np.divide(np.log(fcast_price / price),
#                                    trader.par.risk_aversion * fcast_volatility * price)
#                 return stocks
#
#             try:
#                 p_star_price = scipy.optimize.brentq(optimal_p_star, 0.01, fundamental[-1] * len(traders) * (parameters["init_stocks"] * parameters['fundamental_value'])) #TODO fix bug, what to replace init money with..
#             except:
#                 p_star_price = None #to prevent crashes if the optimizer cannot find the optimal price
#
#             if p_star_price:
#                 p_max = fcast_price
#                 p_min = scipy.optimize.brentq(minimal_p, 0.01, fundamental[-1] * len(traders) * (parameters["init_stocks"] * parameters['fundamental_value']))
#                 trader_price = np.random.uniform(low=p_min, high=p_max)
#                 # get best ask / best bid
#                 lowest_ask_price, highest_bid_price = orderbook.lowest_ask_price, orderbook.highest_bid_price
#
#                 # if the price is lower, than what would make it's current portfolio optimal, the trader buys
#                 if trader_price < p_star_price:
#                     volume = int(min(optimal_stock_holdings(trader_price) - trader.var.stocks, trader.var.money * trader_price))
#                     if volume > 0:
#                         if trader_price >= lowest_ask_price:
#                             bid = orderbook.add_bid(lowest_ask_price, volume, trader)
#                         else:
#                             bid = orderbook.add_bid(trader_price, volume, trader)
#                         trader.var.active_orders.append(bid)
#                 elif trader_price > p_star_price:
#                     volume = int(min(trader.var.stocks - optimal_stock_holdings(trader_price), trader.var.stocks))
#                     if volume > 0:
#                         if trader_price <= highest_bid_price:
#                             ask = orderbook.add_ask(highest_bid_price, volume, trader) # add 'market order' at bid price
#                         else:
#                             ask = orderbook.add_ask(trader_price, volume, trader)
#                         trader.var.active_orders.append(ask)
#
#         # match orders in the order-book
#         while True:
#             matched_orders = orderbook.match_orders()
#             if matched_orders is None:
#                 break
#             # execute trade
#             matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
#             matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])
#
#         # clear and update order-book history
#         orderbook.cleanse_book()
#         orderbook.fundamental = fundamental
#
#     return traders, orderbook