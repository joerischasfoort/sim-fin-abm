from functions.indirect_calibration import *
import time
#import json
import pickle
from multiprocessing import Pool
from SALib.sample import latin
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
  'bounds': [[1, 30], [0.05, 0.30],
             [0.0, 100.0], [0.0, 100.0],
             [0.1, 15.0],
             [100, 300], [0.1, 1.0], [1, 5], [0.1, 0.9],
             [0.1, 1.0]]
}

# use latin hypercube to formulate tuple of all inputs
latin_hyper_cube = latin.sample(problem=problem, N=population_size)
latin_hyper_cube = tuple(latin_hyper_cube.tolist())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]


def optimize(init_set_of_params):
    """Specialized function to run from NM optimizer from 1 starting point"""
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
