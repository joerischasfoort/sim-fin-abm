from init_objects import *
from distribution_model import *
from functions.indirect_calibration import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_value": 166,
              "trader_sample_size": 10, "n_traders": 1000,
              "ticks": 500, "std_fundamental": 0.0530163128919286,
              "std_noise": 0.10696588473846724, "w_random": 1.0,
              "w_fundamentalists": 10.0, "w_momentum": 10.0,
              "init_stocks": 50, "base_risk_aversion": 1.0,
               'spread_max': 0.004087, "horizon": 200}

# 2 initialise model objects TODO perhaps update the diversity of several parameters such as horizon & risk aversion
traders, orderbook = init_objects_distr(parameters, seed=0)

# sim model with calibration
init_params = [22, 0.23362219092236586, 58.56886341766124, 35.18232041845091, 48, 0.3066593229633973, 10]

distr_model_performance(init_params)

# 3 simulate model
traders, orderbook = pb_distr_model(traders, orderbook, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")