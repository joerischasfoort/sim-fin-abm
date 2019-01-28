from functions.indirect_calibration import *
import time
import pickle
from multiprocessing import Pool
import json
import numpy as np

np.seterr(all='ignore')

start_time = time.time()
population_size = 4
CORES = 4 # insert amount of cores available on computer

problem = {
  'num_vars': 10,
  'names': ['trader_sample_size', 'std_noise',
            'w_fundamentalists', 'w_momentum',
            'base_risk_aversion',
            'horizon', "fundamentalist_horizon_multiplier",
            "trades_per_tick", "mutation_probability",
            "average_learning_ability"],
  'bounds': [[2, 30], [0.05, 0.30],
             [0.0, 100.0], [0.0, 100.0],
             [0.1, 15.0],
             [100, 300], [0.1, 1.0], [1, 4], [0.1, 0.9],
             [0.1, 1.0]]
}

with open('test.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]


def optimize(seed):
    """Specialized function to run from NM optimizer from 1 starting point"""
    init_set_of_params = latin_hyper_cube

    return constrNM(distr_model_performance, init_set_of_params, LB, UB, maxiter=15, full_output=True)


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    output = p.map(optimize, latin_hyper_cube) # first argument is function to execute, second argument is tuple of all inputs
    print('All outputs are: ', output)
    # save optimized parameters in pickle file
    with open('nm_calibration_result.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
