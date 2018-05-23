"""Simulation file used to run the model"""

from init_objects import *
from simfinmodel import *

# 1 setup parameters
parameters = {
    # global parameters
    "n_traders": 5000,
    "ticks": 20,
    "trader_sample_size": 4,
    "fundamental_value": 100,
    "std_fundamental": 0.08,
    "std_noise": 0.15,
    "std_vol": 4,
    "max_order_expiration_ticks": 50,
    # trader parameters
    "w_fundamentalists": 0.0,
    "w_momentum": 0.0,
    "w_random": 10.0,
    "w_mean_reversion": 80.0,
    "w_buy_hold": 10.0,
    "spread_max": 0.05, # TODO investigate if this spread is correct
    "horizon_min": 2,
    "horizon_max": 8,
}

# 2 initalise model objects
traders, orderbook = init_objects(parameters)

# 3 simulate model
traders, orderbook = sim_fin_model(traders, orderbook, parameters, seed=1)