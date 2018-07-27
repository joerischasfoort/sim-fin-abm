from functions.stylizedfacts import *
from functions.evolutionaryalgo import *
from functions.helpers import hurst, organise_data
from functions.sensitivity_analysis import *
from SALib.sample import latin
from SALib.sample import saltelli
from SALib.sample.morris import sample
from SALib.analyze import sobol, fast, morris


problem_morris = {
    'num_vars': 13,
    'names': ['n_traders', 'trader_sample_size', 'std_fundamental', 'std_noise',
              'std_vol', 'max_order_expiration_ticks', 'w_fundamentalists', 'w_momentum',
              'w_random', 'w_mean_reversion', 'spread_max',
              'horizon_min', 'horizon_max'],
    'bounds': [[1000, 2000], [1, 30], [0.01, 0.12], [0.05, 0.30],
               [1, 15], [10, 100], [0.0, 100.0], [0.0, 100.0],
               [0.0, 100.0], [0.0, 100.0], [0.01, 0.15], # change mean reversion to 100.0
               [1, 8], [9, 30]]
}

morris_params = sample(problem_morris, 5, num_levels=4, grid_jump=2)
morris_parameter_list = morris_params.tolist()


# convert nescessary parameters to ints
for idx, parameters in enumerate(morris_parameter_list):
    # ints: 0, 1, 4, 5, 11, 12
    morris_parameter_list[idx][0] = int(morris_parameter_list[idx][0])
    morris_parameter_list[idx][1] = int(morris_parameter_list[idx][1])
    morris_parameter_list[idx][4] = int(morris_parameter_list[idx][4])
    morris_parameter_list[idx][5] = int(morris_parameter_list[idx][5])
    morris_parameter_list[idx][11] = int(morris_parameter_list[idx][11])
    morris_parameter_list[idx][12] = int(morris_parameter_list[idx][12])

all_morris_parameters = []
for parameters in morris_parameter_list:
    pars = {}
    for key, value in zip(problem_morris['names'], parameters):
        pars[key] = value
    all_morris_parameters.append(pars)


fixed_parameters = {"ticks": 100, "fundamental_value": 100, "w_buy_hold": 0.0}

morris_output = simulate_params_sobol(NRUNS=1, parameter_set=all_morris_parameters, fixed_parameters=fixed_parameters)
no_autoc = np.array(morris_output['autocorrelation'])
f_tails = np.array(morris_output['kurtosis'])
clustered_vol = np.array(morris_output['autocorrelation_abs'])
l_memory = np.array(morris_output['hurst'])
deviation_from_fundamentals = np.array(morris_output['av_dev_from_fund'])