import random
import numpy as np
import scipy.optimize


def distr_model(traders, orderbook, parameters, seed=1):
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

    for tick in range(parameters['horizon_max'] + 1, parameters["ticks"]):
        # evolve the fundamental value via random walk process
        fundamental.append(fundamental[-1] + parameters["std_fundamental"] * np.random.randn())

        # select random sample of active traders
        active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

        # update common expectation components
        mid_price = orderbook.tick_close_price[-1]
        fundamental_component = np.log(fundamental[-1] / mid_price) #TODO add expected mean reversion time period? as in equation 1
        chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns)-1:-1]
                                       ) / np.arange(1., float(len(orderbook.returns) + 1)) #TODO, did I correctly to calculate horizon max here?

        for trader in active_traders:
            # 1 cancel any active orders
            if trader.var.active_orders:
                for order in trader.var.active_orders:
                    orderbook.cancel_order(order)
                trader.var.active_orders = []

            # update trader specific expectations
            noise_component = parameters['std_noise'] * np.random.randn()

            fcast_return = trader.var.forecast_adjust * (
                trader.var.weight_fundamentalist * fundamental_component +
                trader.var.weight_chartist * chartist_component[trader.par.horizon] +
                trader.var.weight_random * noise_component)

            fcast_price = mid_price * np.exp(fcast_return)
            fcast_volatility = np.var(orderbook.returns[-trader.par.horizon:]) # TODO debug this NEW element

            # update trader specific risk aversion
            trader_risk_aversion = parameters["base_risk_aversion"] * np.divide(1 + trader.var.weight_fundamentalist, #TODO debug this NEW element (equation 6)
                                                                                1 + trader.var.weight_chartist)

            def optimal_p_star(price): #TODO debug this NEW element
                """Determine the price at which the hfm would be satisfied with its current portfolio (eq 10)"""
                price = abs(price)
                stocks = np.divide(np.log(fcast_price / price),
                                   trader_risk_aversion * fcast_volatility * price) - trader.var.stocks
                return stocks

            def minimal_p(price): #TODO debug this NEW element
                price = abs(price)
                stocks_minus_cash = np.divide(np.log(fcast_price / price),
                                   trader_risk_aversion * fcast_volatility * price
                                              ) - trader.var.stocks - trader.var.money
                return stocks_minus_cash

            def optimal_stock_holdings(price): #TODO debug this NEW element
                """Determine the number of stocks a trader wants to hold at a given price (eq 8)"""
                stocks = np.divide(np.log(fcast_price / price),
                                   trader_risk_aversion * fcast_volatility * price)
                return stocks

            # determine optimal holding of stocks TODO debug this NEW element
            try:
                p_star_price = float(scipy.optimize.broyden1(optimal_p_star, mid_price, line_search='wolfe'))
            except:
                p_star_price = None #to prevent crashes if the optimizer cannot find the optimal price

            if p_star_price:
                p_max = fcast_price
                p_min = float(scipy.optimize.broyden1(minimal_p, mid_price, line_search='wolfe')) #TODO debug this NEW function
                trader_price = np.random.uniform(low=p_min, high=p_max)
                # get best ask / best bid
                lowest_ask_price, highest_bid_price = orderbook.lowest_ask_price, orderbook.highest_bid_price

                # if the price is lower, than what would make it's current portfolio optimal, the trader buys
                if trader_price < p_star_price:
                    volume = int(min(optimal_stock_holdings(trader_price) - trader.var.stocks, trader.var.money * trader_price))
                    if volume > 0:
                        if trader_price <= lowest_ask_price:
                            bid = orderbook.add_bid(trader_price, volume, trader) #add 'market order' at ask price TODO debug NEW
                        else:
                            bid = orderbook.add_bid(trader_price, volume, trader)
                        trader.var.active_orders.append(bid)
                elif trader_price > p_star_price:
                    volume = int(min(trader.var.stocks - optimal_stock_holdings(trader_price), trader.var.stocks))
                    if volume > 0:
                        if trader_price >= highest_bid_price:
                            ask = orderbook.add_ask(highest_bid_price, volume, trader) # add 'market order' at bid price
                        else:
                            ask = orderbook.add_ask(trader_price, volume, trader)
                        trader.var.active_orders.append(ask)

        # match orders in the order-book
        while True:
            matched_orders = orderbook.match_orders()
            if matched_orders is None:
                break

        # clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook