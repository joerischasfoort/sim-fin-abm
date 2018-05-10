"""Simulation file used to run the model"""

from init_objects import *
from simfinmodel import *

# 1 setup parameters
parameters = {
    # global parameters
    "n_traders": 5000,
    "ticks": 1000,
    "fundamental_value": 100,
    "std_fundamental": 0.1,
    "std_noise": 0.01,
    "std_vol": 4,
    "max_order_expiration_ticks": 30,
    # trader parameters
    "w_fundamentalists": 0.0,
    "w_momentum": 0.0,
    "w_random": 0.0,
    "w_mean_reversion": 1.0,
    "w_buy_hold": 0.0,
    "spread_max": 0.004087, # from Riordann & Storkenmaier 2012
    # initial values
    "horizon_min": 1,
    "horizon_max": 4,
    "av_return_interval_max": 4,
    "init_spread": (1, 1),
}

# 2 initalise model objects
traders, orderbook = init_objects(parameters)

# 3 simulate model
traders, orderbook = sim_fin_model(traders, orderbook, parameters, seed=1)