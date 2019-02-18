import pandas as pd
import numpy as np
import math
import random
from multiprocessing import Pool
from functions.find_bubbles import *
import time
from functions.stylizedfacts import *
import scipy.stats as stats


np.seterr(all='ignore')

start_time = time.time()

# arguments
BOOTSTRAPS = 1
CORES = 1

shiller_data = pd.read_excel('http://www.econ.yale.edu/~shiller/data/ie_data.xls', header=7)[:-3]

p = pd.Series(np.array(shiller_data.iloc[1174:-1]['Price']))  # starting in 1952  was 1224
y = pd.Series(np.array(shiller_data.iloc[1174:-1]['CAPE']))

obs = len(y)
r0 = 0.01 + 1.8 / np.sqrt(obs)
swindow0 = int(math.floor(r0 * obs))
dim = obs - swindow0 + 1
IC = 2
adflag = 6
yr = 2
Tb = 12 * yr + swindow0 - 1
nboot = 99

def hypothetical_series(starting_value, returns):
    """
    input: starting_value: float starting value
    input: returns: list
    """
    returns = list(returns)
    simulated_series = [starting_value]
    for idx in range(len(returns)):
        simulated_series.append(simulated_series[-1] * (1 + returns[idx]))
    return simulated_series


def bootstrap_bubble_moments(pds):
    pds = pd.Series(pds)

    obs = len(pds)
    dim = obs - swindow0 + 1

    bsadfs = PSY(pds, swindow0, IC, adflag)
    quantilesBsadf = cvPSYwmboot(pds, swindow0, IC, adflag, Tb, nboot)
    monitorDates = pds.iloc[swindow0 - 1:obs].index
    quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
    ind95 = (bsadfs.T[0] > quantile95[1,])
    periods = monitorDates[ind95]

    bubbly_dates = find_sequences_ints(periods, monitorDates)

    perc_bubble_occur = len(periods) / float(len(monitorDates))
    lenghts_of_bubbles = []
    for row in range(len(bubbly_dates)):
        lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)
    av_lenghts_of_bubbles = np.mean(lenghts_of_bubbles)
    stdev_lenghts_bubbles = np.std(lenghts_of_bubbles)
    skews_lenghts_bubbles = pd.Series(lenghts_of_bubbles).skew()
    kurt_lengths_bubbles = pd.Series(lenghts_of_bubbles).kurtosis()

    return [perc_bubble_occur, av_lenghts_of_bubbles, stdev_lenghts_bubbles, skews_lenghts_bubbles, kurt_lengths_bubbles]


def pool_handler():


    bsadfs = PSY(y, swindow0, IC, adflag)

    quantilesBsadf = cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot)
    monitorDates = y.iloc[swindow0 - 1:obs].index
    quantile95 = np.dot(np.array([quantilesBsadf]).T, np.ones([1, dim]))
    ind95 = (bsadfs.T[0] > quantile95[1,])
    periods = monitorDates[ind95]

    bubbly_dates = find_sequences_ints(periods, monitorDates)

    percentage_bubbles = len(periods) / float(len(monitorDates))
    lenghts_of_bubbles = []
    for row in range(len(bubbly_dates)):
        lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)

    av_lenght_bubbles = np.mean(lenghts_of_bubbles)
    stdev_lenght_bubbles = np.std(lenghts_of_bubbles)

    p_returns = pd.Series(np.array(shiller_data.iloc[1174:]['Price'])).pct_change()[1:]
    pd_returns = pd.Series(np.array(shiller_data.iloc[1174:]['CAPE'])).pct_change()[1:]

    emp_moments = np.array([
        autocorrelation_returns(p_returns, 25),
        autocorrelation_abs_returns(p_returns, 25),
        kurtosis(p_returns),
        percentage_bubbles,
        av_lenght_bubbles,
        stdev_lenght_bubbles,
        pd.Series(lenghts_of_bubbles).skew(),
        pd.Series(lenghts_of_bubbles).kurtosis()
    ])

    shiller_block_size = 75  # 8 blocks

    # divide the pd returns into blocks
    pd_data_blocks = []
    p_data_blocks = []
    for x in range(0, len(pd_returns[:-3]), shiller_block_size):
        pd_data_blocks.append(pd_returns[x:x + shiller_block_size])
        p_data_blocks.append(p_returns[x:x + shiller_block_size])

    # draw 5000 random series
    bootstrapped_pd_series = []
    bootstrapped_p_returns = []
    for i in range(BOOTSTRAPS):
        # bootstrap p returns
        sim_data_p = [random.choice(p_data_blocks) for _ in p_data_blocks]
        sim_data2_p = [j for i in sim_data_p for j in i]
        bootstrapped_p_returns.append(sim_data2_p)

        # first sample the data for pd returns
        sim_data = [random.choice(pd_data_blocks) for _ in pd_data_blocks]  # choose a random set of blocks
        sim_data_fundamental_returns = [pair for pair in sim_data]

        # merge the list of lists
        sim_data_fundamental_returns1 = [item for sublist in sim_data_fundamental_returns for item in sublist]

        # calculate the new time_series
        sim_data_fundamentals = hypothetical_series(y[0], sim_data_fundamental_returns1[1:])
        bootstrapped_pd_series.append(sim_data_fundamentals)


    p = Pool(CORES) # argument is how many process happening in parallel
    output = p.map(bootstrap_bubble_moments, bootstrapped_pd_series)  # first argument is function to execute, second argument is tuple of all inputs

    outpt = zip(*output)

    perc_bubble_occur = outpt[0]
    av_lenghts_of_bubbles = outpt[1]
    stdev_lenghts_bubbles = outpt[2]
    skews_lenghts_bubbles = outpt[3]
    kurt_lengths_bubbles = outpt[4]
    # replace NaN value of skew and kurtosis by zero (it is possible there were not enough bubbles to calc these so I assume a normal distribution)
    skews_lenghts_bubbles = list(
        pd.Series(skews_lenghts_bubbles).fillna(0.0))  # np.nanmedian(skews_lenghts_bubbles)))
    kurt_lengths_bubbles = list(pd.Series(kurt_lengths_bubbles).fillna(0.0))  # np.nanmedian(kurt_lengths_bubbles)))

    # calculate bootstrapped returns
    first_order_autocors = []
    mean_abs_autocor = []
    kurtoses = []
    for rets in bootstrapped_p_returns:
        first_order_autocors.append(autocorrelation_returns(rets, 25))
        mean_abs_autocor.append(autocorrelation_abs_returns(rets, 25))
        rets = pd.Series(rets)
        kurtoses.append(kurtosis(rets))

    all_bootstrapped_moments = [first_order_autocors,
                                mean_abs_autocor,
                                kurtoses,
                                perc_bubble_occur,
                                av_lenghts_of_bubbles,
                                stdev_lenghts_bubbles,
                                skews_lenghts_bubbles,
                                kurt_lengths_bubbles
                                ]

    # Get the t-critical value**
    def confidence_interval(sample, emp_value):
        """Calculate confidence_interval in sample"""
        z_critical = stats.norm.ppf(q=0.99)
        stdev = pd.Series(sample).std()
        margin_of_error = z_critical * stdev
        confidence_interval = (emp_value - margin_of_error, emp_value + margin_of_error)
        return confidence_interval

    def get_specific_bootstraps_moments(full_series, bootstrap_number):
        """Get a vector with the moments of a specific bootstrap"""
        return np.array([full_series[i][bootstrap_number] for i in range(len(full_series))])

    av_moments = [np.nanmean(x) for x in all_bootstrapped_moments]
    moments_b = [get_specific_bootstraps_moments(all_bootstrapped_moments, n) for n in
                 range(len(bootstrapped_pd_series))]

    # estimate weighting matrix
    W_hat = 1.0 / len(bootstrapped_pd_series) * sum(
        [np.dot(np.array([(mb - av_moments)]).transpose(), np.array([(mb - av_moments)])) for mb in moments_b])

    W = np.linalg.inv(W_hat)

    print('Weighting matrix is: ', W)
    confidence_intervals = [confidence_interval(m, emp) for m, emp in zip(all_bootstrapped_moments, emp_moments)]

    print('Confidence intervals are: ', confidence_intervals)

    scores = [0 for x in moments_b[0]]
    for bootstr in range(len(moments_b)):
        for idx, moment in enumerate(moments_b[bootstr]):
            if moment > confidence_intervals[idx][0] and moment < confidence_intervals[idx][1]:
                scores[idx] += 1

    MCR_bootstrapped_moments = np.array(scores) / (np.ones(len(scores)) * len(moments_b))
    print('MCRs are: ', MCR_bootstrapped_moments)


if __name__ == '__main__':
    print('Letssss goooo')
    pool_handler()

    print("The bootstraps took", time.time() - start_time, "to run")

