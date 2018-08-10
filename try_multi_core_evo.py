from SALib.sample import latin
from functions.stylizedfacts import *
from functions.evolutionaryalgo import *
from functions.helpers import hurst, organise_data
import time
import json

start_time = time.time()

stylized_facts_spy = {'autocorrelation_abs': 0.049914003700974562,
                      'av_dev_from_fund': 1.9880606822750695,
                      'hurst': 0.41423988932290928,
                      'kurtosis': 3.0676490833426238}

problem = {
  'num_vars': 11,
  'names': ['trader_sample_size', 'std_noise',
            'std_vol', 'max_order_expiration_ticks', 'w_fundamentalists', 'w_momentum',
           'w_random', 'w_mean_reversion', 'spread_max',
           'horizon_min', 'horizon_max'],
  'bounds': [[1, 30], [0.05, 0.30],
             [1, 20], [10, 100], [0.0, 100.0], [0.0, 100.0],
             [1.0, 100.0], [0.0, 100.0], [0.01, 0.15], # change mean reversion to 100.0
             [1, 8], [9, 30]]
}

problem_no_mean_reversion = {
  'num_vars': 10,
  'names': ['trader_sample_size', 'std_noise',
            'std_vol', 'max_order_expiration_ticks', 'w_fundamentalists', 'w_momentum',
           'w_random', 'spread_max',
           'horizon_min', 'horizon_max'],
  'bounds': [[1, 30], [0.05, 0.30],
             [1, 20], [10, 100], [0.0, 100.0], [0.0, 100.0],
             [0.0, 100.0], [0.01, 0.15],
             [1, 8], [9, 30]]
}

population_size = 500

latin_hyper_cube = latin.sample(problem=problem, N=population_size)
latin_hyper_cube = latin_hyper_cube.tolist()

hyper_cube_no_mean_reversion = latin.sample(problem=problem_no_mean_reversion, N=population_size)
hyper_cube_no_mean_reversion = hyper_cube_no_mean_reversion.tolist()

# transform some of the normal parameters to integer
for idx, parameters in enumerate(latin_hyper_cube):
    # ints: 0, 2, 3, 9, 10
    latin_hyper_cube[idx][0] = int(latin_hyper_cube[idx][0])
    latin_hyper_cube[idx][2] = int(latin_hyper_cube[idx][2])
    latin_hyper_cube[idx][3] = int(latin_hyper_cube[idx][3])
    latin_hyper_cube[idx][9] = int(latin_hyper_cube[idx][9])
    latin_hyper_cube[idx][10] = int(latin_hyper_cube[idx][10])

# transform some of the no mean_reversion parameters to integer
for idx, parameters in enumerate(hyper_cube_no_mean_reversion):
    # ints: 0, 2, 3, 9, 10
    hyper_cube_no_mean_reversion[idx][0] = int(hyper_cube_no_mean_reversion[idx][0])
    hyper_cube_no_mean_reversion[idx][2] = int(hyper_cube_no_mean_reversion[idx][2])
    hyper_cube_no_mean_reversion[idx][3] = int(hyper_cube_no_mean_reversion[idx][3])
    hyper_cube_no_mean_reversion[idx][8] = int(hyper_cube_no_mean_reversion[idx][8])
    hyper_cube_no_mean_reversion[idx][9] = int(hyper_cube_no_mean_reversion[idx][9])

# create initial population model for model 1
population = []
for parameters in latin_hyper_cube:
    pars = {}
    for key, value in zip(problem['names'], parameters):
        pars[key] = value
    population.append(Individual(pars, [], np.inf))
all_populations = [population]
av_pop_fitness = []

# create init params for model 2
population_no_mean_reversion = []
for parameters in hyper_cube_no_mean_reversion:
    pars = {}
    for key, value in zip(problem_no_mean_reversion['names'], parameters):
        pars[key] = value
    population_no_mean_reversion.append(Individual(pars, [], np.inf))
all_populations_no_mean_reversion = [population_no_mean_reversion]
av_pop_fitness_no_mean_reversion = []

# fixed parameters
fixed_parameters = {"ticks": 1000, "fundamental_value": 396, "w_buy_hold": 0.0,
                    'n_traders': 1000, 'std_fundamental': 206.5151667007161}

fixed_parameters_no_mean_reversion = {"ticks": 1000, "fundamental_value": 396,
                                      "w_buy_hold": 0.0, "w_mean_reversion": 0.0,
                                      'n_traders': 1000, 'std_fundamental': 206.5151667007161}

iterations = 100
NRUNS = 15
CORES = 4


def simulate_individual(individual):
    """Function to simulate one individual per core"""
    parameters = individual.parameters.copy()
    params = fixed_parameters.copy()
    params.update(parameters)

    stylized_facts = {'autocorrelation': np.inf, 'kurtosis': np.inf, 'autocorrelation_abs': np.inf,
                      'hurst': np.inf, 'av_dev_from_fund': np.inf}

    # simulate the model
    obs = []
    for seed in range(NRUNS):
        traders, orderbook = init_objects.init_objects(params, seed)
        traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
        obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs)
    mc_dev_fundamentals = (mc_prices - mc_fundamentals) / mc_fundamentals

    mean_autocor_abs = []
    mean_kurtosis = []
    long_memory = []
    av_deviation_fundamental = []
    for col in mc_returns:
        mean_autocor_abs.append(np.mean(mc_autocorr_abs_returns[col][1:]))
        mean_kurtosis.append(mc_returns[col][2:].kurtosis())
        long_memory.append(hurst(mc_prices[col][2:]))
        av_deviation_fundamental.append(np.mean(mc_dev_fundamentals[col][1:]))

    stylized_facts['kurtosis'] = np.mean(mean_kurtosis)
    stylized_facts['autocorrelation_abs'] = np.mean(mean_autocor_abs)
    stylized_facts['hurst'] = np.mean(long_memory)
    stylized_facts['av_dev_from_fund'] = np.mean(av_deviation_fundamental)

    cost = cost_function(stylized_facts_spy, stylized_facts)
    next_gen_individual = Individual(parameters, stylized_facts, cost)

    return next_gen_individual


def simulate_individual2(individual):
    """Function to simulate one individual per core"""
    parameters = individual.parameters.copy()
    params = fixed_parameters_no_mean_reversion.copy()
    params.update(parameters)

    stylized_facts = {'autocorrelation': np.inf, 'kurtosis': np.inf, 'autocorrelation_abs': np.inf,
                      'hurst': np.inf, 'av_dev_from_fund': np.inf}

    # simulate the model
    obs = []
    for seed in range(NRUNS):
        traders, orderbook = init_objects.init_objects(params, seed)
        traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
        obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs)
    mc_dev_fundamentals = (mc_prices - mc_fundamentals) / mc_fundamentals

    mean_autocor_abs = []
    mean_kurtosis = []
    long_memory = []
    av_deviation_fundamental = []
    for col in mc_returns:
        mean_autocor_abs.append(np.mean(mc_autocorr_abs_returns[col][1:]))
        mean_kurtosis.append(mc_returns[col][2:].kurtosis())
        long_memory.append(hurst(mc_prices[col][2:]))
        av_deviation_fundamental.append(np.mean(mc_dev_fundamentals[col][1:]))

    stylized_facts['kurtosis'] = np.mean(mean_kurtosis)
    stylized_facts['autocorrelation_abs'] = np.mean(mean_autocor_abs)
    stylized_facts['hurst'] = np.mean(long_memory)
    stylized_facts['av_dev_from_fund'] = np.mean(av_deviation_fundamental)

    cost = cost_function(stylized_facts_spy, stylized_facts)
    next_gen_individual = Individual(parameters, stylized_facts, cost)

    return next_gen_individual


def pool_handler():
    p = Pool(CORES)  # argument is how many process happening in parallel
    # first simulate model 1
    for generation in range(iterations):
        # simulate every individual in the population using multiple cores
        simulated_population = p.map(simulate_individual, all_populations[generation])
        # sort population to have fittest left
        simulated_population.sort(key=lambda x: x.cost, reverse=False)
        # calculate average fitness of population
        average_population_fitness = average_fitness(simulated_population)
        # record population fitness
        av_pop_fitness.append(average_population_fitness)
        print('generation: ', generation, 'fitness: ', average_population_fitness)
        # add a new, evolved population to the list of populations
        all_populations.append(evolve_population(simulated_population, fittest_to_retain=0.3, random_to_retain=0.2,
                                                 parents_to_mutate=0.3, parameters_to_mutate=0.1, problem=problem))

    # Then, simulate model 2 (no mean reversion)
    for generation in range(iterations):
        # simulate every individual in the population using multiple cores
        simulated_population = p.map(simulate_individual2, all_populations_no_mean_reversion[generation])
        # sort population to have fittest left
        simulated_population.sort(key=lambda x: x.cost, reverse=False)
        # calculate average fitness of population
        average_pop_fitness_no_mean_reversion = average_fitness(simulated_population)
        # record population fitness
        av_pop_fitness_no_mean_reversion.append(average_pop_fitness_no_mean_reversion)
        print('no_mean_reversion generation: ', generation, 'fitness: ', average_pop_fitness_no_mean_reversion)
        # add a new, evolved population to the list of populations
        all_populations_no_mean_reversion.append(evolve_population(simulated_population, fittest_to_retain=0.3, random_to_retain=0.2,
                                                 parents_to_mutate=0.3, parameters_to_mutate=0.1, problem=problem_no_mean_reversion))



if __name__ == '__main__':
    pool_handler()
    # store best parameters
    parameters = all_populations[-1][0].parameters.copy()
    params = fixed_parameters.copy()
    params.update(parameters)

    parameters2 = all_populations_no_mean_reversion[-1][0].parameters.copy()
    params2 = fixed_parameters.copy()
    params2.update(parameters2)

    best_params = {'model1': params, 'model2': params2}
    with open('best_params.json', 'w') as fp:
        json.dump(best_params, fp)

    # print reproduction of stylized facts
    with open('sim_stylized_facts_model1.json', 'w') as fp:
        json.dump(all_populations[-1][0].stylized_facts, fp)
    with open('sim_stylized_facts_model2.json', 'w') as fp:
        json.dump(all_populations_no_mean_reversion[-1][0].stylized_facts, fp)
    # print evolution of fitness
    ftnss = {'model1': av_pop_fitness, 'model2': av_pop_fitness_no_mean_reversion}
    with open('fitness.json', 'w') as fp:
        json.dump(ftnss, fp)

    print("The simulations took", time.time() - start_time, "to run")