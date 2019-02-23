from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import numpy as np
import math
from functions.find_bubbles import *
import json

np.seterr(all='ignore')

def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    for x in range(0, 17):
        output = sim_synthetic_bubble(x)
    #output = p.map(sim_synthetic_bubble, list_of_seeds)

    with open('shocked_bubbles_output.json', 'w') as fp:
        json.dump(output, fp)

    print('All outputs are: ', output)


if __name__ == '__main__':
    start_time = time.time()

    NRUNS = 24
    CORES = 4  # set the amount of cores equal to the amount of runs

    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
