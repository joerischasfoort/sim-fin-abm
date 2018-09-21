""""This file contains a function to run various simulations for a sobol sensitivity analysis"""
import random
import numpy as np
import simfinmodel
import init_objects
from functions.stylizedfacts import *
from functions.helpers import *
from functions.evolutionaryalgo import m_fitness


def simulate_params_sobol(NRUNS, parameter_set, fixed_parameters):
    """
    Simulate the simfin model for a set of changing and fixed parameters
    :param NRUNS: integer amount of Monte Carlo simulations
    :param parameter_set: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed
    :return: numpy array of average stylized facts outcome values for all parameter combinations
    """

    # stylized facts order: first_order_autocors, autocors1, autocors5, mean_abs_autocor, kurtosis, spy_abs_auto10, spy_abs_auto25, spy_abs_auto50, spy_abs_auto100, cointegrations
    stylized_facts = [[],[],[],[],[],[],[],[],[],[]]

    for parameters in parameter_set:
        # combine individual parameters with fixed parameters
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

        stylized_facts[0].append(np.mean(first_order_autocors))
        stylized_facts[1].append(np.mean(autocors1))
        stylized_facts[2].append(np.mean(autocors5))
        stylized_facts[3].append(np.mean(mean_abs_autocor))
        stylized_facts[4].append(np.mean(kurtoses))
        stylized_facts[5].append(np.mean(spy_abs_auto10))
        stylized_facts[6].append(np.mean(spy_abs_auto25))
        stylized_facts[7].append(np.mean(spy_abs_auto50))
        stylized_facts[8].append(np.mean(spy_abs_auto100))
        stylized_facts[9].append(np.mean(cointegrations))

    return stylized_facts


def sim_robustness(NRUNS, parameter_set, fixed_parameters, empirical_moments, W, confidence_intervals_moments):
    """
    Simulate the simfin model for a set of changing and fixed parameters while outputting the j-score & MCRs
    :param NRUNS: integer amount of Monte Carlo simulations
    :param parameter_set: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed.
    :param empirical_moments: np.Array of empirical moments
    :param W: np.Matrix inverse var covar matrix of bootstrapped data
    :return: dictionary containing the j-score and mcr-scores
    """
    scores = {'j_score': [], 'mcr_scores': []}

    for parameters in parameter_set:
        # combine individual parameters with fixed parameters
        params = fixed_parameters.copy()
        params.update(parameters)

        # simulate the model
        obs = []
        for seed in range(NRUNS):
            traders, orderbook = init_objects.init_objects_contrarians(params, seed)
            traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, params, seed)
            obs.append(orderbook)

        # store simulated stylized facts
        mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
            obs)

        scores['j_score'].append(m_fitness(mc_returns, mc_prices, mc_fundamentals, empirical_moments, W))

        mcr_scores_model = []
        for col in mc_prices:
            mcr_scores_model.append(get_model_moments_in_confidence(pd.DataFrame(mc_returns[col]),
                                                                    pd.DataFrame(mc_prices[col]),
                                                                    pd.DataFrame(mc_fundamentals[col]),
                                                                    confidence_intervals_moments))

        # calc MC scores
        scores['mcr_scores'].append([true_scores(mcr_scores_model, i) for i in range(len(empirical_moments))])

    return scores


def m_core_sim_run(parameters): #TODO update
    """
    Run the simulation once with a set of parameters. Can be used to run on multiple cores
    :param NRUNS: integer number of runs to simulate the model for one instance of parameters
    :param parameters: dictionary of parameters
    :return:
    """
    stylized_facts = {'autocorrelation': [], 'kurtosis': [], 'autocorrelation_abs': [],
                      'hurst': [], 'av_dev_from_fund': []}

    obs = []
    for seed in range(5): # TODO amount of RUNS hardcoded
        traders, orderbook = init_objects.init_objects(parameters, seed)
        traders, orderbook = simfinmodel.sim_fin_model(traders, orderbook, parameters, seed)
        obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs)
    mc_dev_fundamentals = (mc_prices - mc_fundamentals) / mc_fundamentals

    mean_autocor = []
    mean_autocor_abs = []
    mean_kurtosis = []
    long_memory = []
    av_deviation_fundamental = []
    for col in mc_returns:
        mean_autocor.append(np.mean(mc_autocorr_returns[col][1:]))  # correct?
        mean_autocor_abs.append(np.mean(mc_autocorr_abs_returns[col][1:]))
        mean_kurtosis.append(mc_returns[col][2:].kurtosis())
        long_memory.append(hurst(mc_prices[col][2:]))
        av_deviation_fundamental.append(np.mean(mc_dev_fundamentals[col][1:]))

    stylized_facts['autocorrelation'] = (np.mean(mean_autocor))
    stylized_facts['kurtosis'] = (np.mean(mean_kurtosis))
    stylized_facts['autocorrelation_abs'] = (np.mean(mean_autocor_abs))
    stylized_facts['hurst'] = (np.mean(long_memory))
    stylized_facts['av_dev_from_fund']= (np.mean(av_deviation_fundamental))

    return stylized_facts
