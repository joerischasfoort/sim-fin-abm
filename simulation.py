"""Simulation file used to run the model"""

from init_objects import *
from simfinmodel import *

# # 1 setup parameters
# parameters = {
#     # global parameters set through evolutionary algorithm calibration
#     "n_traders": 1309,
#     "ticks": 1000,
#     "trader_sample_size": 21,
#     "fundamental_value": 100,
#     "std_fundamental": 0.08317728524869135,
#     "std_noise": 0.05633716087190844,
#     "std_vol": 5,
#     "max_order_expiration_ticks": 18,
#     # trader parameters
#     "w_fundamentalists": 20.84484458217016,
#     "w_momentum": 58.107737854582844,
#     "w_random": 1.4948772502086316,
#     "w_mean_reversion": 69.1289833734435,
#     "w_buy_hold": 0.0,
#     "spread_max": 0.044494473036647685,
#     "horizon_min": 7,
#     "horizon_max": 11,
# }

parameters = {'w_buy_hold': 0.0, 'max_order_expiration_ticks': 30,
              'w_random': 75.03175032570262, 'n_traders': 1000,
              'horizon_min': 4, 'trader_sample_size': 28,
              'fundamental_value': 396, 'w_fundamentalists': 60.485968301189594,
              'spread_max': 0.08518940388804527, 'w_momentum': 83.73547977223258,
              'horizon_max': 10, 'std_vol': 19, 'w_mean_reversion': 91.62137028108113,
              'std_fundamental': 206.5151667007161, 'ticks': 1000, 'std_noise': 0.22068770505861335}

# 2 initalise model objects
traders, orderbook = init_objects(parameters, seed=0)

# 3 simulate model
traders, orderbook = sim_fin_model(traders, orderbook, parameters, seed=0)