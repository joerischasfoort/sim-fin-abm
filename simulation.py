"""Simulation file used to run the model"""

from init_objects import *
from simfinmodel import *

# 1 setup parameters
parameters = {
    # global parameters set through evolutionary algorithm calibration
    "n_traders": 5000,
    "ticks": 20,
    "trader_sample_size": 26,
    "fundamental_value": 100,
    "std_fundamental": 0.10872321803799265,
    "std_noise": 0.20508407891045205,
    "std_vol": 1,
    "max_order_expiration_ticks": 50,
    # trader parameters
    "w_fundamentalists": 76.66646857709344,
    "w_momentum": 47.53421946123167,
    "w_random": 1.4948772502086316,
    "w_mean_reversion": 65.01917050983388,
    "w_buy_hold": 10.0,
    "spread_max": 0.040136915087013045,
    "horizon_min": 6,
    "horizon_max": 15,
}

# 2 initalise model objects
traders, orderbook = init_objects(parameters, seed=1)

# 3 simulate model
traders, orderbook = sim_fin_model(traders, orderbook, parameters, seed=1)