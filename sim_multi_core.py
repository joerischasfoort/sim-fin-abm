"""Simulation file used to run the model on multiple cores"""

from functions.sensitivity_analysis import m_core_sim_run
from multiprocessing import Pool
import time

start_time = time.time()
CORES = 1 # insert amount of cores available on computer

parameters = {
    # global parameters set through evolutionary algorithm calibration
    "n_traders": 1309,
    "ticks": 1000,
    "trader_sample_size": 21,
    "fundamental_value": 100,
    "std_fundamental": 0.08317728524869135,
    "std_noise": 0.05633716087190844,
    "std_vol": 5,
    "max_order_expiration_ticks": 18,
    # trader parameters
    "w_fundamentalists": 20.84484458217016,
    "w_momentum": 58.107737854582844,
    "w_random": 1.4948772502086316,
    "w_mean_reversion": 69.1289833734435,
    "w_buy_hold": 0.0,
    "spread_max": 0.044494473036647685,
    "horizon_min": 7,
    "horizon_max": 11,
}

parameter_set = [parameters for x in range(4)]


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    output = p.map(m_core_sim_run, parameter_set) # first argument is function to execute, second argument is tuple of all inputs
    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")