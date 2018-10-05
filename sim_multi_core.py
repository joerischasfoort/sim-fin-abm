"""Simulation file used to run the model on multiple cores"""

from functions.sensitivity_analysis import m_core_sim_run
from multiprocessing import Pool
import time

start_time = time.time()
CORES = 1 # insert amount of cores available on computer

parameters = {'max_order_expiration_ticks': 30,
              'w_random': 75.03175032570262, 'n_traders': 1000,
              'trader_sample_size': 28,
              'fundamental_value': 396, 'w_fundamentalists': 60.485968301189594,
              'spread_max': 0.08518940388804527, 'w_momentum': 83.73547977223258,
              'horizon_max': 10, 'std_vol': 19, 'w_mean_reversion': 91.62137028108113,
              'std_fundamental': 0.05151667007161, 'ticks': 1000, 'std_noise': 0.22068770505861335}

parameter_set = [parameters for x in range(4)]


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    output = p.map(m_core_sim_run, parameter_set) # first argument is function to execute, second argument is tuple of all inputs
    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")