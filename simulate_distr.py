from init_objects import *
from distribution_model import *
import time

start_time = time.time()

# # 1 setup parameters
parameters = {"fundamental_value": 166,
              "trader_sample_size": 10, "n_traders": 5000,
              "ticks": 1000, "std_fundamental": 0.0530163128919286,
              "std_noise": 0.10696588473846724,
              "w_fundamentalists": 10.0, "w_momentum": 0.0,
              "init_stocks": 25, "base_risk_aversion": 0.05,
              "w_random": 1.0,
              "horizon": 200}

# 2 initalise model objects
traders, orderbook = init_objects_distr(parameters, seed=0)

# 3 simulate model
traders, orderbook = distr_model(traders, orderbook, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")