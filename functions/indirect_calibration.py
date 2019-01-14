from distribution_model import pb_distr_model
from init_objects import init_objects_distr
from functions.helpers import organise_data
from functions.evolutionaryalgo import *


def distr_model_performance(input_parameters):
    """
    Simple function calibrate uncertain model parameters
    :param input_parameters: list of input parameters
    :return: cost
    """
    problem = {
        'num_vars': 7,
        'names': ['trader_sample_size', 'std_noise',
                  'w_fundamentalists', 'w_momentum',
                  'init_stocks', 'base_risk_aversion',
                  'horizon'],
        'bounds': [[1, 30], [0.05, 0.30],
                   [0.0, 100.0], [0.0, 100.0],
                   [1, 100], [0.1, 15.0],
                   [9, 30]]
    }

    # update params
    uncertain_parameters = dict(zip(problem['names'], input_parameters))
    params = {"ticks": 1235, "fundamental_value": 166, 'n_traders': 1000, 'std_fundamental': 0.0530163128919286,
              'spread_max': 0.004087, "w_random": 1.0}
    params.update(uncertain_parameters)

    empirical_moments = np.array([-9.56201354e-03, -9.55051841e-02, -5.52010512e-02,
                                  3.35217232e-01, 1.24673150e+01, 3.46352635e-01,
                                  2.72135459e-01, 1.88193342e-01, 1.75876698e-01])

    traders = []
    obs = []
    # run model with parameters
    for seed in range(5):  # TODO update fixed runs?
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

    stylized_facts_sim = np.array([
        np.mean(first_order_autocors),
        np.mean(autocors1),
        np.mean(autocors5),
        np.mean(mean_abs_autocor),
        np.mean(kurtoses),
        np.mean(spy_abs_auto10),
        np.mean(spy_abs_auto25),
        np.mean(spy_abs_auto50),
        np.mean(spy_abs_auto100)
    ])

    # calculate the cost
    cost = quadratic_loss_function(stylized_facts_sim, empirical_moments, np.identity(len(stylized_facts_sim)))
    return cost
