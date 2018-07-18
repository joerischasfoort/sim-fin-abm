""""This file contains functions which can be used for an evolutionary algorithm"""
import random
import bisect
import numpy as np
import simfinmodel
import init_objects
from functions.stylizedfacts import *
from statistics import mean
from functions.helpers import hurst


def average_fitness(population):
    total_cost = 0
    for individual in population:
        total_cost += individual.cost
    return total_cost / (float(len(population)))


def cost_function(observed_values, average_simulated_values):
    """cost function"""
    score = 0
    for obs, sim in zip(observed_values, average_simulated_values):
        score += ((obs - sim) / obs)**2
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

    # 3 mutate random parameters of random individuals TODO add here to only mutate mutable parameters
    amount_of_individuals_to_mutate = int(parents_to_mutate * len(parents))
    amount_of_params_to_mutate = int(parameters_to_mutate * len(parents[0].parameters))
    for parent in random.sample(parents, amount_of_individuals_to_mutate):
        indexes_of_mutable_params = random.sample(range(len(parent.parameters)), amount_of_params_to_mutate)
        for idx in indexes_of_mutable_params:
            min_value, max_value = problem['bounds'][idx][0], problem['bounds'][idx][1]
            if type(min_value) == float:
                parent.parameters[idx] = random.uniform(min_value, max_value)
            else:
                parent.parameters[idx] = random.randint(min_value, max_value)

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
            child_parameters = male.parameters[:half] + female.parameters[half:]
            child = Individual(child_parameters, [], np.inf)
            children.append(child)
    parents.extend(children)
    # the parents list now contains a full new population with the parents and their offspring
    return parents


def simulate_population(population, NRUNS, stylized_facts_real_life): #TODO debug
    """
    Simulate a population of parameter spaces for the sim-fin model
    :param population: population of parameter spaces used to simulate model
    :param number_of_runs: number of times the simulation should be run
    :param simulation_time: amount of days which will be simulated for each run
    :return: simulated population, average population fitness
    """
    simulated_population = []
    for idx, individual in enumerate(population):
        parameters = individual.parameters
        stylized_facts = {'autocorrelation': np.inf, 'kurtosis': np.inf, 'autocorrelation_abs': np.inf,
                          'hurst': np.inf, 'hurst_dev_from_fund': np.inf}

        # simulate the model
        #traders = []
        obs = []
        for seed in range(NRUNS):
            traders, orderbook = init_objects.init_objects(parameters, seed)
            traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, parameters, seed)
            # traders.append(traders)
            obs.append(orderbook)

            # store simulated stylized facts
            mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
                obs)
            mc_dev_fundaments = mc_prices - mc_fundamentals

            mean_kurtosis = []
            long_memory = []
            long_memory_deviation_fundamentals = []
            for col in mc_returns:
                kurtosis = mc_returns[col][2:].kurtosis()
                lm = hurst(mc_prices[col][2:])
                mean_kurtosis.append(kurtosis)
                long_memory.append(lm)
                long_memory_deviation_fundamentals.append(hurst(mc_dev_fundaments[col][2:]))

            stylized_facts['autocorrelation'] = mc_autocorr_returns.mean(axis=1)
            stylized_facts['kurtosis'] = np.mean(mean_kurtosis)
            stylized_facts['autocorrelation_abs'] = mc_autocorr_abs_returns.mean(axis=1)
            stylized_facts['hurst'] = np.mean(lm)
            stylized_facts['hurst_dev_from_fund'] = np.mean(long_memory_deviation_fundamentals)

        # create next generation individual
        next_gen_individual = Individual(parameters, [], np.inf)
        # add average stylized facts to individual
        for s_fact in stylized_facts:
            next_gen_individual.stylized_facts.append(mean(stylized_facts[s_fact]))
        # add average fitness to individual
        next_gen_individual.cost = cost_function(stylized_facts_real_life, next_gen_individual.stylized_facts)
        # insert into next generation population, lowest score to the left
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