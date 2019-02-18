from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import json
import numpy as np
import math
from functions.find_bubbles import *

np.seterr(all='ignore')

start_time = time.time()

# INPUT PARAMETERS
LATIN_NUMBER = 1
NRUNS = 5
BURN_IN = 400
CORES = NRUNS # set the amount of cores equal to the amount of runs

problem = {
  'num_vars': 7,
  'names': ['std_noise',
            'w_fundamentalists', 'w_momentum',
            'base_risk_aversion',
            "fundamentalist_horizon_multiplier",
            "mutation_probability",
            "average_learning_ability"],
  'bounds': [[0.05, 0.30],
             [0.0, 100.0], [0.0, 100.0],
             [0.1, 15.0],
             [0.1, 1.0], [0.1, 0.9],
             [0.1, 1.0]]
}

with open('hypercube.txt', 'r') as f:
    latin_hyper_cube = json.loads(f.read())

# Bounds
LB = [x[0] for x in problem['bounds']]
UB = [x[1] for x in problem['bounds']]

init_parameters = latin_hyper_cube[LATIN_NUMBER]

params = {"ticks": 1200 + BURN_IN, "fundamental_value": 166, 'n_traders': 500, 'std_fundamental': 0.0530163128919286,
          'spread_max': 0.004087, "w_random": 1.0, "init_stocks": 50, 'trader_sample_size': 19,
          'horizon': 200, "trades_per_tick": 2}


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
        obs, burn_in_period=BURN_IN)

    obs = len(mc_fundamentals[0])
    r0 = 0.01 + 1.8 / np.sqrt(obs)
    swindow0 = int(math.floor(r0 * obs))
    dim = obs - swindow0 + 1
    IC = 2
    adflag = 6
    yr = 2
    Tb = 12 * yr + swindow0 - 1
    nboot = 99

    first_order_autocors = []
    mean_abs_autocor = []
    kurtoses = []
    perc_bubble_occur = []
    av_lenghts_of_bubbles = []
    stdev_lenghts_bubbles = []
    skews_lenghts_bubbles = []
    kurt_lengths_bubbles = []

    for idx, col in enumerate(mc_returns):
        first_order_autocors.append(autocorrelation_returns(mc_returns[col][1:], 25))
        mean_abs_autocor.append(autocorrelation_abs_returns(mc_returns[col][1:], 25))
        kurtoses.append(mc_returns[col][2:].kurtosis())
        # calc bubble stats
        pds = pd.Series(mc_prices[idx][:-1] / mc_fundamentals[idx])

        obs = len(pds)
        dim = obs - swindow0 + 1

        bsadfs = PSY(pds, swindow0, IC, adflag)
        quantilesBsadf = cvPSYwmboot(pds, swindow0, IC, adflag, Tb, nboot)
        monitorDates = pds.iloc[swindow0 - 1:obs].index
        quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
        ind95 = (bsadfs.T[0] > quantile95[1,])
        periods = monitorDates[ind95]

        # only proceed with calculating bubble statistics if there were any
        if True in ind95:
            bubbly_dates = find_sequences_ints(periods, monitorDates)

            perc_bubble_occur.append(len(periods) / float(len(monitorDates)))
            lenghts_of_bubbles = []
            for row in range(len(bubbly_dates)):
                lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)
            av_lenghts_of_bubbles.append(np.mean(lenghts_of_bubbles))
            stdev_lenghts_bubbles.append(np.std(lenghts_of_bubbles))
            skews_lenghts_bubbles.append(pd.Series(lenghts_of_bubbles).skew())
            kurt_lengths_bubbles.append((pd.Series(lenghts_of_bubbles).kurtosis()))

        else:
            perc_bubble_occur.append(np.inf)
            av_lenghts_of_bubbles.append(np.inf)
            stdev_lenghts_bubbles.append(np.inf)
            skews_lenghts_bubbles.append(np.inf)
            kurt_lengths_bubbles.append(np.inf)

    # replace NaN value of skew and kurtosis by zero (it is possible there were not enough bubbles to calc these so I assume a normal distribution)
    skews_lenghts_bubbles = list(pd.Series(skews_lenghts_bubbles).fillna(0.0))
    kurt_lengths_bubbles = list(pd.Series(kurt_lengths_bubbles).fillna(0.0))

    stylized_facts_sim = np.array([np.mean(first_order_autocors),
                                   np.mean(mean_abs_autocor),
                                   np.mean(kurtoses),
                                   np.mean(perc_bubble_occur),
                                   np.mean(av_lenghts_of_bubbles),
                                   np.mean(stdev_lenghts_bubbles),
                                   np.mean(skews_lenghts_bubbles),
                                   np.mean(kurt_lengths_bubbles)
                                   ])

    W = np.load('distr_weighting_matrix.npy')  # if this doesn't work, use: np.identity(len(stylized_facts_sim))

    empirical_moments = np.array([0.0094336,  0.05371445,  2.67297082,  0.08876812,  6.125,
                                  5.34877322,  0.83048364, -0.91551026])

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
        variable_names = ['std_noise', 'w_fundamentalists', 'w_momentum',
                          'base_risk_aversion', "fundamentalist_horizon_multiplier",
                          "mutation_probability", "average_learning_ability"]

        # convert relevant parameters to integers
        new_input_params = []
        for idx, par in enumerate(input_parameters):
            new_input_params.append(par)

        # update params
        uncertain_parameters = dict(zip(variable_names, new_input_params))
        params = {"ticks": 600 + BURN_IN, "fundamental_value": 166, 'n_traders': 500, 'std_fundamental': 0.0530163128919286,
                  'spread_max': 0.004087, "w_random": 1.0, "init_stocks": 50, 'trader_sample_size': 19,
                  'horizon': 200, "trades_per_tick": 2}  # TODO make ticks: 600 * 10
        params.update(uncertain_parameters)

        list_of_seeds_params = [[seed, params] for seed in list_of_seeds]

        # costs = []
        # for seed_par in list_of_seeds_params:
        #     costs.append(simulate_a_seed(seed_par))

        costs = p.map(simulate_a_seed, list_of_seeds_params) # first argument is function to execute, second argument is tuple of all inputs TODO uncomment this

        return np.mean(costs)

    output = constrNM(model_performance, init_parameters, LB, UB, maxiter=3, full_output=True)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
