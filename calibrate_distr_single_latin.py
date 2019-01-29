from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import json
import numpy as np

np.seterr(all='ignore')

start_time = time.time()

# INPUT PARAMETERS
LATIN_NUMBER = 0
NRUNS = 4
CORES = NRUNS # set the amount of cores equal to the amount of runs

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

with open('hypercube.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]

init_parameters = latin_hyper_cube[LATIN_NUMBER]

params = {"ticks": 25160, "fundamental_value": 166, 'n_traders': 500, 'std_fundamental': 0.0530163128919286,
                  'spread_max': 0.004087, "w_random": 1.0, "init_stocks": 50}  # TODO make ticks: 2516 * 10


def simulate_a_seed(seed_params):
    """Simulates the model for a single seed and outputs the associated cost"""
    seed = seed_params[0]
    params = seed_params[1]

    traders = []
    obs = []
    # run model with parameters
    traders, orderbook = init_objects_distr(params, seed)
    traders, orderbook = pb_distr_model(traders, orderbook, params, seed)
    traders.append(traders)
    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs)

    first_order_autocors = []
    autocors1 = []
    autocors5 = []
    mean_abs_autocor = []
    kurtoses = []
    spy_abs_auto10 = []
    spy_abs_auto25 = []
    spy_abs_auto50 = []
    spy_abs_auto100 = []
    spy_abs_auto150 = []
    spy_abs_auto200 = []
    for col in mc_returns:
        first_order_autocors.append(autocorrelation_returns(mc_returns[col][1:], 25))
        autocors1.append(mc_returns[col][1:].autocorr(lag=1))
        autocors5.append(mc_returns[col][1:].autocorr(lag=5))
        mean_abs_autocor.append(autocorrelation_abs_returns(mc_returns[col][1:], 25))
        kurtoses.append(mc_returns[col][2:].kurtosis())
        spy_abs_auto10.append(mc_returns[col][1:].abs().autocorr(lag=10))
        spy_abs_auto25.append(mc_returns[col][1:].abs().autocorr(lag=25))
        spy_abs_auto50.append(mc_returns[col][1:].abs().autocorr(lag=50))
        spy_abs_auto100.append(mc_returns[col][1:].abs().autocorr(lag=100))
        spy_abs_auto150.append(mc_returns[col][1:].abs().autocorr(lag=150))
        spy_abs_auto200.append(mc_returns[col][1:].abs().autocorr(lag=200))

    stylized_facts_sim = np.array([
        np.mean(first_order_autocors),
        np.mean(autocors1),
        np.mean(autocors5),
        np.mean(mean_abs_autocor),
        np.mean(kurtoses),
        np.mean(spy_abs_auto10),
        np.mean(spy_abs_auto25),
        np.mean(spy_abs_auto50),
        np.mean(spy_abs_auto100),
        np.mean(spy_abs_auto150),
        np.mean(spy_abs_auto200)
    ])

    W = np.load('distr_weighting_matrix.npy')  # if this doesn't work, use: np.identity(len(stylized_facts_sim))

    empirical_moments = np.array([-7.91632942e-03, -6.44109792e-02, -5.17149408e-02, 2.15757804e-01,
                                  4.99915089e+00, 2.29239806e-01, 1.36705815e-01, 8.99171488e-02, 3.97109985e-02,
                                  4.56905198e-02, 3.40685479e-03])

    # calculate the cost
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    return cost


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    def model_performance(input_parameters):
        """
        Simple function calibrate uncertain model parameters
        :param input_parameters: list of input parameters
        :return: average cost
        """
        integer_var_locations = [0, 5, 7]
        variable_names = ['trader_sample_size', 'std_noise', 'w_fundamentalists', 'w_momentum',
                          'base_risk_aversion', 'horizon', "fundamentalist_horizon_multiplier",
                          "trades_per_tick", "mutation_probability", "average_learning_ability"]

        # convert relevant parameters to integers
        new_input_params = []
        for idx, par in enumerate(input_parameters):
            if idx in integer_var_locations:
                new_input_params.append(int(par))
            else:
                new_input_params.append(par)

        # update params
        uncertain_parameters = dict(zip(variable_names, new_input_params))
        params = {"ticks": 25160, "fundamental_value": 166, 'n_traders': 500, 'std_fundamental': 0.0530163128919286,
                  'spread_max': 0.004087, "w_random": 1.0, "init_stocks": 50}  # TODO make ticks: 2516 * 10
        params.update(uncertain_parameters)

        list_of_seeds_params = [[seed, params] for seed in list_of_seeds]
        costs = p.map(simulate_a_seed, list_of_seeds_params) # first argument is function to execute, second argument is tuple of all inputs

        return np.mean(costs)

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=1, full_output=True)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
