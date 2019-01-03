from init_objects import *
from distribution_model import *
import time

start_time = time.time()

# # 1 setup parameters TODO add parameters
parameters = {"spread_max": 0.004087, "fundamental_value": 166,
              "trader_sample_size": 19, "n_traders": 5000,
              "ticks": 25000, "std_fundamental": 0.0530163128919286,
              "std_noise": 0.10696588473846724,
              "w_fundamentalists": 8.489180919376432, "w_momentum": 43.055017297045524,
              "init_money": 8300, "init_stocks": 50, "base_risk_aversion": 0.1,
              "max_order_expiration_ticks": 30, "std_vol": 7, "w_random": 73.28414619497076,
              "horizon": 200}

# 2 initalise model objects
traders, orderbook = init_objects_distr(parameters, seed=0)

# 3 simulate model
traders, orderbook = distr_model(traders, orderbook, parameters, seed=0)

print("The simulations took", time.time() - start_time, "to run")