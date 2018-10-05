"""Simulation file used to run the model"""

from init_objects import *
from simfinmodel import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {'max_order_expiration_ticks': 30,
              'w_random': 75.03175032570262, 'n_traders': 1000,
              'trader_sample_size': 28,
              'fundamental_value': 396, 'w_fundamentalists': 60.485968301189594,
              'spread_max': 0.08518940388804527, 'w_momentum': 83.73547977223258,
              'horizon_max': 10, 'std_vol': 19, 'w_mean_reversion': 91.62137028108113,
              'std_fundamental': 0.05151667007161, 'ticks': 1000, 'std_noise': 0.22068770505861335}

# 2 initalise model objects
traders, orderbook = init_objects(parameters, seed=0)

# 3 simulate model
traders, orderbook = sim_fin_model(traders, orderbook, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")