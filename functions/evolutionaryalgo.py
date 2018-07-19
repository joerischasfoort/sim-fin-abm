""""This file contains functions which can be used for an evolutionary algorithm"""
import random
import bisect
import numpy as np
import simfinmodel
import init_objects
from functions.stylizedfacts import *
from statistics import mean
from functions.helpers import *


def average_fitness(population):
    total_cost = 0
    for individual in population:
        total_cost += individual.cost
    return total_cost / (float(len(population)))


def cost_function(observed_values, average_simulated_values):
    """
    Calculate average squared deviation of simulated values from observed values
    :param observed_values: dictionary of observed stylized facts
    :param average_simulated_values: dictionary of corresponding simulated stylized facts
    :return:
    """
    score = 0
    for key in observed_values:
        score += ((observed_values[key] - average_simulated_values[key]) / observed_values[key])**2
    return score


def evolve_population(population, fittest_to_retain, random_to_retain, parents_to_mutate, parameters_to_mutate, problem): #TODO change
    """
    Evolves a population. First, the fittest members of the population plus some random individuals become parents.
    Then, some random mutations take place in the parents. Finally, the parents breed to create children.
    :param population: population individuals sorted by cost (cheapest left) which contain parameter values
    :param fittest_to_retain: percentage of fittest individuals which should be maintained as parents
    :param random_to_retain: percentage of other random individuals which should be maintained as parents
    :param individuals_to_mutate: percentage of parents in which mutations will take place
    :param parameters_to_mutate: percentage of parameters in chosen individuals which will mutate
    :return:
    """
    # 1 retain parents
    retain_lenght = int(len(population) * fittest_to_retain)
    parents = population[:retain_lenght]

    # 2 retain random individuals
    amount_random_indiv = int(len(population) * random_to_retain)
    parents.extend(random.sample(population[retain_lenght:], amount_random_indiv))

    if not parents:
        raise ValueError('There are no parents, so evolution cannot take place')

    # 3 mutate random parameters of random individuals
    amount_of_individuals_to_mutate = int(parents_to_mutate * len(parents))
    amount_of_params_to_mutate = int(parameters_to_mutate * len(parents[0].parameters))
    for parent in random.sample(parents, amount_of_individuals_to_mutate):
        indexes_of_mutable_params = random.sample(range(len(parent.parameters)), amount_of_params_to_mutate)
        for idx in indexes_of_mutable_params:
            min_value, max_value = problem['bounds'][idx][0], problem['bounds'][idx][1]
            key = problem['names'][idx]
            if type(min_value) == float:
                parent.parameters[key] = random.uniform(min_value, max_value)
            else:
                parent.parameters[key] = random.randint(min_value, max_value)

    # 4 parents breed to create a new population
    parents_lenght = len(parents)
    desired_lenght = len(population) - parents_lenght
    children = []
    if parents_lenght < 2:
        raise ValueError('There are not enough parents to breed, increase either fittest_to_retain or random_to_retain')
    while len(children) < desired_lenght:
        male = random.randint(0, parents_lenght - 1)
        female = random.randint(0, parents_lenght - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male.parameters) / 2)
            # use the problem['names'] here to find the correct parameters TODO shuffle parameters
            child_parameters = {}
            param_names = random.sample(problem['names'], len(problem['names']))
            male_params = param_names[:half]
            for key in male_params:
                child_parameters[key] = male.parameters[key]

            female_params = param_names[half:]
            for key in female_params:
                child_parameters[key] = female.parameters[key]
            #child_parameters = male.parameters[:half] + female.parameters[half:]
            child = Individual(child_parameters, [], np.inf)
            children.append(child)
    parents.extend(children)
    # the parents list now contains a full new population with the parents and their offspring
    return parents


def simulate_population(population, NRUNS, fixed_parameters, stylized_facts_real_life): #TODO debug
    """
    Simulate a population of parameter spaces for the sim-fin model
    :param population: population of parameter spaces used to simulate model
    :param number_of_runs: number of times the simulation should be run
    :param simulation_time: amount of days which will be simulated for each run
    :param fixed_parameters: dictionary of parameters which will not be changed
    :return: simulated population, average population fitness
    """
    simulated_population = []
    for idx, individual in enumerate(population):
        # combine individual parameters with fixed parameters
        parameters = individual.parameters.copy()
        params = fixed_parameters.copy()
        params.update(parameters)

        stylized_facts = {'autocorrelation': np.inf, 'kurtosis': np.inf, 'autocorrelation_abs': np.inf,
                          'hurst': np.inf, 'hurst_dev_from_fund': np.inf}

        # simulate the model
        #traders = []
        obs = []
        for seed in range(NRUNS):
            traders, orderbook = init_objects.init_objects(params, seed)
            traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
            # traders.append(traders)
            obs.append(orderbook)

        # store simulated stylized facts
        mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(obs)
        mc_dev_fundaments = mc_prices - mc_fundamentals

        mean_autocor = []
        mean_autocor_abs = []
        mean_kurtosis = []
        long_memory = []
        long_memory_deviation_fundamentals = []
        for col in mc_returns:
            mean_autocor.append(np.mean(mc_autocorr_returns[col][1:])) #correct?
            mean_autocor_abs.append(np.mean(mc_autocorr_abs_returns[col][1:]))
            mean_kurtosis.append(mc_returns[col][2:].kurtosis())
            long_memory.append(hurst(mc_prices[col][2:]))
            long_memory_deviation_fundamentals.append(hurst(mc_dev_fundaments[col][2:]))

        stylized_facts['autocorrelation'] = np.mean(mean_autocor) # TODO correct?
        stylized_facts['kurtosis'] = np.mean(mean_kurtosis)
        stylized_facts['autocorrelation_abs'] = np.mean(mean_autocor_abs) #TODO correct?
        stylized_facts['hurst'] = np.mean(long_memory)
        stylized_facts['hurst_dev_from_fund'] = np.mean(long_memory_deviation_fundamentals)

        # create next generation individual
        cost = cost_function(stylized_facts_real_life, stylized_facts) # TODO debug
        next_gen_individual = Individual(parameters, stylized_facts, cost) # TODO check if no copy mistakes
        # insert into next generation population, lowest score (best) to the left
        bisect.insort_left(simulated_population, next_gen_individual)

    average_population_fitness = average_fitness(simulated_population)

    return simulated_population, average_population_fitness


class Individual:
    """The order class can represent both bid or ask type orders"""
    def __init__(self, parameters, stylized_facts, cost):
        self.parameters = parameters
        self.stylized_facts = stylized_facts
        self.cost = cost

    def __lt__(self, other):
        """Allows comparison to other individuals based on its cost (negative fitness)"""
        return self.cost < other.cost