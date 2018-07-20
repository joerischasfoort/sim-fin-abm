import pandas as pd
from SALib.sample import latin
from functions.stylizedfacts import *
from functions.evolutionaryalgo import *
from functions.helpers import hurst, organise_data


stylized_facts_spy = {'autocorrelation_abs': 0.049914003700974562, 'av_dev_from_fund': 2.0192681168484445,
                      'hurst': 0.41423988932290989, 'kurtosis': 3.0676490833426238}

problem = {
  'num_vars': 13,
  'names': ['n_traders', 'trader_sample_size', 'std_fundamental', 'std_noise',
            'std_vol', 'max_order_expiration_ticks', 'w_fundamentalists', 'w_momentum',
           'w_random', 'w_mean_reversion', 'spread_max',
           'horizon_min', 'horizon_max'],
  'bounds': [[1000, 2000], [1, 30], [0.01, 0.12], [0.05, 0.30],
             [1, 15], [10, 100], [0.0, 100.0], [0.0, 100.0],
             [0.0, 100.0], [0.0, 100.0], [0.01, 0.15],
             [1, 8], [9, 30]]
}

population_size = 30
latin_hyper_cube = latin.sample(problem=problem, N=population_size)
latin_hyper_cube = latin_hyper_cube.tolist()

# transform some of the parameters to integer
for idx, parameters in enumerate(latin_hyper_cube):
    # ints: 0, 1, 4, 5, 11, 12
    latin_hyper_cube[idx][0] = int(latin_hyper_cube[idx][0])
    latin_hyper_cube[idx][1] = int(latin_hyper_cube[idx][1])
    latin_hyper_cube[idx][4] = int(latin_hyper_cube[idx][4])
    latin_hyper_cube[idx][5] = int(latin_hyper_cube[idx][5])
    latin_hyper_cube[idx][11] = int(latin_hyper_cube[idx][11])
    latin_hyper_cube[idx][12] = int(latin_hyper_cube[idx][12])

# create initial population
population = []
for parameters in latin_hyper_cube:
    pars = {}
    for key, value in zip(problem['names'], parameters):
        pars[key] = value
    population.append(Individual(pars, [], np.inf))
all_populations = [population]
av_pop_fitness = []

fixed_parameters = {"ticks": 1000, "fundamental_value": 100, "w_buy_hold": 0.0}
iterations = 12
NRUNS = 5

for generation in range(iterations):
    # simulate every population
    simulated_population, fitness = simulate_population(all_populations[generation], NRUNS, fixed_parameters, stylized_facts_spy)
    # record population fitness
    av_pop_fitness.append(fitness)
    print('generation: ', generation, 'fitness: ', fitness)
    # add a new, evolved population to the list of populations
    all_populations.append(evolve_population(simulated_population, fittest_to_retain=0.3, random_to_retain=0.2,
                                             parents_to_mutate=0.3, parameters_to_mutate=0.1, problem=problem))