""""The main model"""
import numpy as np
import random


def sim_fin_model(traders, orderbook, parameters, seed=1):
    """The main model function"""
    random.seed(seed)
    np.random.seed(seed)

    for tick in range(parameters['av_return_interval_max'] + 1, parameters["ticks"]):

        # select active traders
        active_traders = [traders[np.random.randint(1, len(traders))]]

        # update common price components
        mid_price = orderbook.tick_close_price[-1]
        fundamental_component = np.log(parameters['fundamental_value'] / mid_price)
        noise_component = parameters['std_noise'] * np.random.randn()
        chartist_component = np.cumsum(orderbook.returns[-parameters['av_return_interval_max']:]
                                       ) / np.arange(1., float(parameters['av_return_interval_max'] + 1))

        for trader in active_traders:
            # update expectations
            fcast_return = trader.var.forecast_adjust * (
                trader.var.weight_fundamentalist * fundamental_component +
                trader.var.weight_chartist * chartist_component[trader.par.horizon] +
                trader.var.weight_random * noise_component -
                trader.var.weight_mean_reversion * chartist_component[trader.par.horizon] +
                trader.var.weight_buy_hold * 0.0)
            fcast_return = min(fcast_return, 0.5)
            fcast_return = max(fcast_return, -0.5)
            fcast_price = mid_price * np.exp(fcast_return)
            # submit orders
            if fcast_price > mid_price:
                bid_price = fcast_price * (1. - trader.par.spread)
                orderbook.add_bid(bid_price, abs(int(np.random.normal(scale=parameters['std_vol']))), trader)
            elif fcast_price < mid_price:
                ask_price = fcast_price * (1 + trader.par.spread)
                orderbook.add_ask(ask_price, abs(int(np.random.normal(scale=parameters['std_vol']))), trader)

        while True:
            matched_orders = orderbook.match_orders()
            if matched_orders is None:
                break

        orderbook.cleanse_book()

    return traders, orderbook
