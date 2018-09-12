from SALib.sample import latin
from functions.evolutionaryalgo import *
from functions.helpers import organise_data
import json
import time

start_time = time.time()

empirical_moments = np.array([ -9.56201354e-03,  -9.55051841e-02,  -5.52010512e-02,
         3.35217232e-01,   1.24673150e+01,   3.46352635e-01,
         2.72135459e-01,   1.88193342e-01,   1.75876698e-01,
        -3.39594806e+00])

# Inverse of estimated covariance matrix of the bootstrapped data moments
W = np.array([[  3.19813864e+05,  -1.23647688e+04,  -7.75253249e+02,
         -9.96776357e+02,  -3.46059595e+01,  -8.45442951e+01,
          1.30642711e+02,   8.21228832e+02,  -7.53254275e+02,
          4.01244860e-01],
       [ -1.23647688e+04,   3.09643983e+03,  -9.12387428e+02,
          7.15088809e+02,   4.00622557e+00,  -2.07088870e+02,
          2.23333826e+02,   2.47454265e+01,  -7.69804083e+01,
         -9.32556575e-01],
       [ -7.75253249e+02,  -9.12387428e+02,   2.45450034e+03,
         -5.66285993e+02,   2.50019644e+00,   7.52973406e+01,
         -6.53708966e+01,   1.49542112e+01,  -3.22682344e+01,
         -8.15971301e-02],
       [ -9.96776357e+02,   7.15088809e+02,  -5.66285993e+02,
          6.03184170e+02,  -6.93622662e+00,  -3.92832732e+01,
          5.68213213e+01,  -2.51354209e+01,  -5.92040756e-01,
         -1.47482933e-01],
       [ -3.46059595e+01,   4.00622557e+00,   2.50019644e+00,
         -6.93622662e+00,   2.12483995e-01,  -1.07699983e+00,
          7.97173695e-01,   6.64963370e-01,  -4.66126569e-01,
          1.13807341e-03],
       [ -8.45442951e+01,  -2.07088870e+02,   7.52973406e+01,
         -3.92832732e+01,  -1.07699983e+00,   1.23523697e+04,
         -1.37167420e+04,  -1.80945248e+03,   5.03115269e+03,
         -5.47676063e-01],
       [  1.30642711e+02,   2.23333826e+02,  -6.53708966e+01,
          5.68213213e+01,   7.97173695e-01,  -1.37167420e+04,
          1.68349608e+04,  -6.34482252e+01,  -5.39173033e+03,
          4.87711333e-01],
       [  8.21228832e+02,   2.47454265e+01,   1.49542112e+01,
         -2.51354209e+01,   6.64963370e-01,  -1.80945248e+03,
         -6.34482252e+01,   5.75725915e+03,  -4.01464585e+03,
         -3.56169092e-01],
       [ -7.53254275e+02,  -7.69804083e+01,  -3.22682344e+01,
         -5.92040756e-01,  -4.66126569e-01,   5.03115269e+03,
         -5.39173033e+03,  -4.01464585e+03,   5.75734730e+03,
          5.83832248e-01],
       [  4.01244860e-01,  -9.32556575e-01,  -8.15971301e-02,
         -1.47482933e-01,   1.13807341e-03,  -5.47676063e-01,
          4.87711333e-01,  -3.56169092e-01,   5.83832248e-01,
          3.93494501e-01]])

# simulation time = 10 * T (where T is the lenght of the empirical data
simulation_time = 2500 # * 10

problem = {
  'num_vars': 8,
  'names': ['trader_sample_size', 'std_noise',
            'std_vol', 'w_fundamentalists', 'w_momentum',
           'w_random', 'w_mean_reversion',
           'horizon_max'],
  'bounds': [[5, 20], [0.05, 0.15],
             [4, 10], [0.0, 100.0], [0.0, 100.0],
             [1.0, 100.0], [0.0, 100.0],
             [5, 30]]
}

# problem for the model without mean reversion
problem_nmr = {
  'num_vars': 7,
  'names': ['trader_sample_size', 'std_noise',
            'std_vol', 'w_fundamentalists', 'w_momentum',
           'w_random', 'horizon_max'],
  'bounds': [[5, 20], [0.05, 0.15],
             [4, 10], [0.0, 100.0], [0.0, 100.0],
             [1.0, 100.0], [5, 30]]
}

# population size for evolutionary algo
population_size = 50

# create init paramters for both models
latin_hyper_cube = latin.sample(problem=problem, N=population_size)
latin_hyper_cube = latin_hyper_cube.tolist()

for idx, parameters in enumerate(latin_hyper_cube):
    # ints: 0, 2, 7
    latin_hyper_cube[idx][0] = int(latin_hyper_cube[idx][0])
    latin_hyper_cube[idx][2] = int(latin_hyper_cube[idx][2])
    latin_hyper_cube[idx][7] = int(latin_hyper_cube[idx][7])

latin_hyper_cube_nmr = latin.sample(problem=problem_nmr, N=population_size)
latin_hyper_cube_nmr = latin_hyper_cube_nmr.tolist()

for idx, parameters in enumerate(latin_hyper_cube_nmr):
    # ints: 0, 2, 6
    latin_hyper_cube_nmr[idx][0] = int(latin_hyper_cube_nmr[idx][0])
    latin_hyper_cube_nmr[idx][2] = int(latin_hyper_cube_nmr[idx][2])
    latin_hyper_cube_nmr[idx][6] = int(latin_hyper_cube_nmr[idx][6])

# create initial populations for both models

population = []
for parameters in latin_hyper_cube:
    pars = {}
    for key, value in zip(problem['names'], parameters):
        pars[key] = value
    population.append(Individual(pars, [], np.inf))
all_populations = [population]
av_pop_fitness = []

population_nmr = []
for parameters in latin_hyper_cube_nmr:
    pars = {}
    for key, value in zip(problem_nmr['names'], parameters):
        pars[key] = value
    population_nmr.append(Individual(pars, [], np.inf))
all_populations_nmr = [population_nmr]
av_pop_fitness_nmr = []

# determine fixed parameters (see notebook)
start_fundamental_value = 166
std_fundamental_value = 0.0530163128919286
#burn_in_period = 100 should I use this?

fixed_parameters = {"ticks": simulation_time, "fundamental_value": start_fundamental_value,
                    'n_traders': 1000, 'std_fundamental': std_fundamental_value, 'spread_max': 0.004087,
                    'max_order_expiration_ticks': 30}

fixed_parameters_nmr = {"ticks": simulation_time, "fundamental_value": start_fundamental_value,
                    'n_traders': 1000, 'std_fundamental': std_fundamental_value, 'spread_max': 0.004087,
                    'max_order_expiration_ticks': 30, "w_mean_reversion": 0.0}

iterations = 40
NRUNS = 2
CORES = 4

def simulate_individual(individual):
    """Function to simulate one individual per core for model"""
    # combine individual parameters with fixed parameters
    parameters = individual.parameters.copy()
    params = fixed_parameters.copy()
    params.update(parameters)

    # simulate the model
    obs = []
    for seed in range(NRUNS):
        traders, orderbook = init_objects.init_objects(params, seed)
        traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
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
    cointegrations = []
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
        cointegrations.append(cointegr(mc_prices[col][1:], mc_fundamentals[col][1:])[0])

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
        np.mean(cointegrations)
    ])

    # create next generation individual
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    next_gen_individual = Individual(parameters, stylized_facts_sim, cost)

    return next_gen_individual


def simulate_individual_nmr(individual):
    """Function to simulate one individual per core for the no mean reversion model"""
    # combine individual parameters with fixed parameters
    parameters = individual.parameters.copy()
    params = fixed_parameters_nmr.copy()
    params.update(parameters)

    # simulate the model
    obs = []
    for seed in range(NRUNS):
        traders, orderbook = init_objects.init_objects(params, seed)
        traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
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
    cointegrations = []
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
        cointegrations.append(cointegr(mc_prices[col][1:], mc_fundamentals[col][1:])[0])

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
        np.mean(cointegrations)
    ])

    # create next generation individual
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, W)
    next_gen_individual = Individual(parameters, stylized_facts_sim, cost)

    return next_gen_individual


def pool_handler():
    """Main function to be able to simulate on multiple cores"""
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
        simulated_population = p.map(simulate_individual_nmr, all_populations_nmr[generation])
        # sort population to have fittest left
        simulated_population.sort(key=lambda x: x.cost, reverse=False)
        # calculate average fitness of population
        average_pop_fitness_no_mean_reversion = average_fitness(simulated_population)
        # record population fitness
        av_pop_fitness_nmr.append(average_pop_fitness_no_mean_reversion)
        print('no_mean_reversion generation: ', generation, 'fitness: ', average_pop_fitness_no_mean_reversion)
        # add a new, evolved population to the list of populations
        all_populations_nmr.append(evolve_population(simulated_population, fittest_to_retain=0.3, random_to_retain=0.2,
                                                     parents_to_mutate=0.3, parameters_to_mutate=0.1,
                                                     problem=problem_nmr))

if __name__ == '__main__':
    pool_handler()
    # store best parameters
    parameters = all_populations[-1][0].parameters.copy()
    params = fixed_parameters.copy()
    params.update(parameters)

    parameters2 = all_populations_nmr[-1][0].parameters.copy()
    params2 = fixed_parameters.copy()
    params2.update(parameters2)

    best_params = {'model1': params, 'model2': params2}
    with open('best_params.json', 'w') as fp:
        json.dump(best_params, fp)

    # print reproduction of stylized facts
    with open('sim_stylized_facts_model1.json', 'w') as fp:
        json.dump(all_populations[-1][0].stylized_facts.tolist(), fp)
    with open('sim_stylized_facts_model2.json', 'w') as fp:
        json.dump(all_populations_nmr[-1][0].stylized_facts.tolist(), fp)
    # print evolution of fitness
    ftnss = {'model1': av_pop_fitness, 'model2': av_pop_fitness_nmr}
    with open('fitness.json', 'w') as fp:
        json.dump(ftnss, fp)

    print("The simulations took", time.time() - start_time, "to run")

