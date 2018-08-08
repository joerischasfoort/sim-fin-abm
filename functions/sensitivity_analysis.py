""""This file contains a function to run various simulations for a sobol sensitivity analysis"""
import random
import numpy as np
import simfinmodel
import init_objects
from functions.stylizedfacts import *
from functions.helpers import *


def simulate_params_sobol(NRUNS, parameter_set, fixed_parameters):
    """
    Simulate the simfin model for a set of changing and fixed parameters
    :param stylized_fact: string parameter. Either: 'autocorrelation', 'kurtosis', 'autocorrelation_abs', 'hurst' or 'av_dev_from_fund'
    :param NRUNS: integer amount of Monte Carlo simulations
    :param population: list of parameters which have been sampled for Sobol sensitivity analysis
    :param fixed_parameters: list of parameters which will remain fixed
    :return: numpy array of average stylized facts outcome values for all parameter combinations
    """
    stylized_facts = {'autocorrelation': [], 'kurtosis': [], 'autocorrelation_abs': [],
                      'hurst': [], 'av_dev_from_fund': []}

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

        stylized_facts['autocorrelation'].append(np.mean(mean_autocor))
        stylized_facts['kurtosis'].append(np.mean(mean_kurtosis))
        stylized_facts['autocorrelation_abs'].append(np.mean(mean_autocor_abs))
        stylized_facts['hurst'].append(np.mean(long_memory))
        stylized_facts['av_dev_from_fund'].append(np.mean(av_deviation_fundamental))

    return stylized_facts


def m_core_sim_run(parameters):
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
