from functions.indirect_calibration import *
import time
from multiprocessing import Pool
import json
import numpy as np
import math
from functions.find_bubbles import *
from functions.inequality import *

np.seterr(all='ignore')

start_time = time.time()

NRUNS = 4
CORES = 4 # set the amount of cores equal to the amount of runs


def sim_bubble_info(seed):
    """
    Simulate model once and return accompanying info on
    - bubble_type
    - bubble-episode price
    - wealth_start
    - wealth_end
    + wealth_gini_over_time
    + palma_over_time
    + twentytwenty_over_time
    """
    BURN_IN = 400
    with open('parameters.json', 'r') as f:
        params = json.loads(f.read())
    # simulate model once
    #traders = []
    obs = []
    # run model with parameters
    traders, orderbook = init_objects_distr(params, seed)
    traders, orderbook = pb_distr_model(traders, orderbook, params, seed)
    #traders.append(traders)
    obs.append(orderbook)

    # store simulated stylized facts
    mc_prices, mc_returns, mc_autocorr_returns, mc_autocorr_abs_returns, mc_volatility, mc_volume, mc_fundamentals = organise_data(
        obs, burn_in_period=BURN_IN)

    y = pd.Series(mc_prices[0][:-1] / mc_fundamentals[0])

    obs = len(y)
    r0 = 0.01 + 1.8 / np.sqrt(obs)
    swindow0 = int(math.floor(r0 * obs))
    dim = obs - swindow0 + 1
    IC = 2
    adflag = 6
    yr = 2
    Tb = 12 * yr + swindow0 - 1
    nboot = 99

    # calc bubbles
    bsadfs = PSY(y, swindow0, IC, adflag)

    quantilesBsadf = cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot=99)

    monitorDates = y.iloc[swindow0 - 1:obs].index
    quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
    ind95 = (bsadfs.T[0] > quantile95[1,])
    periods = monitorDates[ind95]

    bubble_types = []
    bubble_prices = []
    wealth_starts = []
    wealth_ends = []
    ginis_ot = []
    palmas_ot = []
    twtws_ot = []

    if True in ind95:
        bubbly_dates = find_sequences_ints(periods, monitorDates)
        proper_bubbles = bubbly_dates.iloc[p_bubbles(bubbly_dates)]

        # classify the bubbles
        start_dates = []
        end_dates = []
        for l in range(len(proper_bubbles)):
            start_dates.append(proper_bubbles.iloc[l]['start_date'])
            end_dates.append(proper_bubbles.iloc[l]['end_date'])

            if abs(y[end_dates[l]] - y[start_dates[l]]) > y[:end_dates[l]].std():
                # classify as boom or bust
                if y[start_dates[l]] > y[end_dates[l]]:
                    bubble_type = 'bust'
                else:
                    bubble_type = 'boom'
            else:
                if y[start_dates[l]:end_dates[l]].mean() > y[start_dates[l]]:
                    # classify as boom-bust or bust-boom
                    bubble_type = 'boom-bust'
                else:
                    bubble_type = 'bust-boom'
            bubble_types.append(bubble_type)

            # determine the start and end wealth of the bubble
            money_start = np.array([x.var.money[BURN_IN + start_dates[l]] for x in traders])
            stocks_start = np.array([x.var.stocks[BURN_IN + start_dates[l]] for x in traders])
            wealth_start = money_start + (stocks_start * mc_prices[0].iloc[start_dates[l]])

            money_end = np.array([x.var.money[BURN_IN + end_dates[l]] for x in traders])
            stocks_end = np.array([x.var.stocks[BURN_IN + end_dates[l]] for x in traders])
            wealth_end = money_end + (stocks_end * mc_prices[0].iloc[end_dates[l]])

            wealth_gini_over_time = []
            palma_over_time = []
            twentytwenty_over_time = []
            for t in range(BURN_IN + start_dates[l], BURN_IN + end_dates[l]):
                money = np.array([x.var.money[t] for x in traders])
                stocks = np.array([x.var.stocks[t] for x in traders])
                wealth = money + (stocks * orderbook.tick_close_price[t])

                share_top_10 = sum(np.sort(wealth)[int(len(wealth) * 0.9):]) / sum(wealth)
                share_bottom_40 = sum(np.sort(wealth)[:int(len(wealth) * 0.4)]) / sum(wealth)
                palma_over_time.append(share_top_10 / share_bottom_40)

                share_top_20 = np.mean(np.sort(wealth)[int(len(wealth) * 0.8):])
                share_bottom_20 = np.mean(np.sort(wealth)[:int(len(wealth) * 0.2)])
                twentytwenty_over_time.append(share_top_20 / share_bottom_20)

                wealth_gini_over_time.append(gini(wealth))

            bubble_prices.append(list(mc_prices[0].iloc[start_dates[l]: end_dates[l]]))
            wealth_starts.append(list(wealth_start))
            wealth_ends.append(list(wealth_end))
            ginis_ot.append(wealth_gini_over_time)
            palmas_ot.append(palma_over_time)
            twtws_ot.append(twentytwenty_over_time)

    return bubble_types, bubble_prices, wealth_starts, wealth_ends, ginis_ot, palmas_ot, twtws_ot


def pool_handler():
    p = Pool(CORES) # argument is how many process happening in parallel
    list_of_seeds = [x for x in range(NRUNS)]

    output = p.map(sim_bubble_info, list_of_seeds)

    with open('many_bubbles_output.json', 'w') as fp:
        json.dump(output, fp)

    print('All outputs are: ', output)


if __name__ == '__main__':
    pool_handler()
    print("The simulations took", time.time() - start_time, "to run")
