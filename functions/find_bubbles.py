import numpy as np
import pandas as pd
import math


def ADF(y, IC=0, adflag=0):
    """
    Calculates the augmented Dickey-Fuller (ADF) test statistic with lag order set fixed or selected by AIC or BIC

    Port from github psymonitor
    Credits to: Phillips, P C B, Shi, S, & Yu, J
    Testing for multiple bubbles: Historical episodes of exuberance and collapse in the SP 500
    International Economic Review 2015

    :param y: list data
    :param IC:
    :param adflag:
    :return: float ADF test statistic
    """
    T0 = len(y)
    T1 = len(y) - 1
    const = np.ones(T1)

    y1 = np.array(y[0:T1])
    y0 = np.array(y[1:T0])
    dy = y0 - y1
    x1 = np.c_[y1, const]

    t = T1 - adflag
    dof = t - adflag - 2

    if IC > 0:
        ICC = np.zeros([adflag + 1, 1])
        ADF = np.zeros([adflag + 1, 1])
        for k in range(adflag + 1):
            dy01 = dy[k:T1, ]
            x2 = np.zeros([T1 - k, k])

            for j in range(k):
                x2[:, j] = dy[k - j - 1:T1 - j - 1]

            x2 = np.concatenate((x1[k:T1, ], x2,), axis=1)

            # OLS regression
            beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
            eps = dy01 - np.dot(x2, beta)
            # Information criteria
            npdf = sum(-1 / 2.0 * np.log(2 * np.pi) - 1 / 2.0 * (eps ** 2))
            if IC == 1:
                ICC[k, ] = -2 * npdf / float(t) + 2 * len(beta) / float(t)
            elif IC == 2:
                ICC[k, ] = -2 * npdf / float(t) + len(beta) * np.log(t) / float(t)
            se = np.dot(eps.T, eps / dof)
            sig = np.sqrt(np.diag((np.ones([len(beta), len(beta)]) * se) * np.linalg.solve(np.dot(x2.T, x2),
                                                                                           np.identity(
                                                                                               len(np.dot(x2.T, x2))))))
            ADF[k,] = beta[0,] / sig[0]
        lag0 = np.argmin(ICC)
        ADFlag = ADF[lag0,][0]  # TODO check if this is correct
    elif IC == 0:
        # Model Specification
        dy01 = dy[adflag:T1, ]
        x2 = np.zeros([t, adflag])

        for j in range(adflag):
            x2[:, j] = dy[adflag - j - 1:T1 - j - 1]

        x2 = np.concatenate((x1[adflag:T1, ], x2,), axis=1)

        # OLS regression
        beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
        eps = dy01 - np.dot(x2, beta)
        se = np.dot(eps.T, eps / dof)
        sig = np.sqrt(np.diag((np.ones([len(beta), len(beta)]) * se) * np.linalg.solve(np.dot(x2.T, x2), np.identity(
            len(np.dot(x2.T, x2))))))

        ADFlag = beta[0,] / sig[0]  # check if this is correct

    if IC == 0:
        result = ['fixed lag of order 1', ADFlag]

    if IC == 1:
        result = ['ADF Statistic using AIC', ADFlag]

    if IC == 2:
        result = ['ADF Statistic using BIC', ADFlag]

    return result[1]


def PSY(y, swindow0, IC, adflag):
    """
    Estimate PSY's BSADF sequence of test statistics
    implements the real time bubble detection procedure of Phillips, Shi and Yu (2015a,b)

    param: y: np.array of the data
    param: swindow0: integer minimum window size
    param: adflag: An integer, lag order when IC=0; maximum number of lags when IC>0 (default = 0).

    For every period in time calculate the max ADF statistic using a rolling window.

    return: list BSADF test statistic.
    """
    t = len(y)

    if not swindow0:
        swindow0 = int(math.floor(t * (0.01 + 1.8 / np.sqrt(t))))

    bsadfs = np.zeros([t, 1])  # create empty column array at lenght of the data (zeros)

    for r2 in range(swindow0, t + 1):
        # loop over the range 47 - 647
        # create column vector of increasing lenght and fill with - 999
        rwadft = np.ones([r2 - swindow0 + 1, 1]) * -999
        for r1 in range(r2 - swindow0 + 1):
            # loop over the range 0 - 500
            # perform ADF test on data from r1 --> r2
            # insert in row
            rwadft[r1] = float(ADF(y.iloc[r1:r2], IC, adflag))

        # take max value an insert in bsadfs array
        bsadfs[r2 - 1] = max(rwadft.T[0])

    # create shortened version of array
    bsadf = bsadfs[swindow0-1 : t]

    return bsadf


def ADFres(y, IC=0, adflag=0):
    """"""
    T0 = len(y)
    T1 = len(y) - 1

    y1 = np.array(y[0:T1])
    y0 = np.array(y[1:T0])
    dy = y0 - y1
    t = T1 - adflag

    if IC > 0:
        ICC = np.zeros([adflag + 1, 1])
        betaM = []  # np.zeros([adflag + 1, 1])
        epsM = []  # np.zeros([adflag + 1, 1])
        for k in range(adflag + 1):
            dy01 = dy[k:T1, ]
            x_temp = np.zeros([T1 - k, k])

            for j in range(k):
                x_temp[:, j] = dy[k - j - 1:T1 - j - 1]

            x2 = np.ones([T1 - k, k + 1])
            x2[:, 1:] = x_temp

            # OLS regression time
            betaM.append(
                np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01)))
            epsM.append(dy01 - np.dot(x2, betaM[k]))

            # Information criteria
            npdf = sum(-1 / 2.0 * np.log(2 * np.pi) - 1 / 2.0 * (epsM[k] ** 2))
            if IC == 1:
                ICC[k] = -2 * npdf / float(t) + 2 * len(betaM[k]) / float(t)
            elif IC == 2:
                ICC[k] = -2 * npdf / float(t) + len(betaM[k]) * np.log(t) / float(t)

        lag0 = np.argmin(ICC)
        beta = betaM[lag0]
        eps = epsM[lag0]
        lag = lag0

    elif IC == 0:
        dy01 = dy[adflag:T1, ]
        x_temp = np.zeros([t, adflag])

        for j in range(adflag):
            x_temp[:, j] = dy[adflag - j - 1:T1 - j - 1]

        x2 = np.ones([T1 - adflag, adflag + 1]) #TODO debug this line
        x2[:, 1:] = x_temp

        # OLS regression
        beta = np.dot(np.linalg.solve(np.dot(x2.T, x2), np.identity(len(np.dot(x2.T, x2)))), np.dot(x2.T, dy01))
        eps = dy01 - np.dot(x2, beta)
        lag = adflag

    else:
        beta, eps, lag = None

    return beta, eps, lag


def cvPSYwmboot(y, swindow0, IC, adflag, Tb, nboot=199):
    """
    Computes a matrix of 90, 95 and 99 critical values which can be used to compare to the bsadf statistics.
    param: y: data array or pandas df
    param: swindow0: integer minimum window size
    param: adflag: An integer, lag order when IC=0; maximum number of lags when IC>0 (default = 0).
    param: control_sample_size: Integer the simulated sample size
    param: nboot: positive integer. Number of bootstrap replications (default = 199).
    param: nCores = integernumber of cores (supports multithreading
    return: A matrix. BSADF bootstrap critical value sequence at the 90, 95 and 99 percent level.
    """
    qe = np.array([0.90, 0.95, 0.99])

    beta, eps, lag = ADFres(y, IC, adflag)

    T0 = len(eps)
    t = len(y)
    dy = np.array(y.iloc[0:(t - 1)]) - np.array(y.iloc[1:t])
    g = len(beta)

    # create matrix filled with random ints < T0 with rows TB and cols: nboot
    rN = np.random.randint(0, T0, (Tb, nboot))
    # create weigth matrix filled a random normal float
    wn = np.random.normal(0) * np.ones([Tb, nboot])

    # dyb = 69 row, 99 col matrix of zeros
    dyb = np.zeros([Tb - 1, nboot])
    # fill first 6 rows with first six
    dyb[:lag + 1, ] = np.split(np.tile(dy[[l for l in range(lag + 1)]], nboot), lag + 1, axis=0)

    for j in range(nboot):
        # loop over all columns
        if lag == 0:
            for i in range(lag, Tb - 1):
                # loop over rows, start filling the rest of the dyb rows with random numbers
                dyb[i, j] = wn[i - lag, j] * eps[rN[i - lag, j]]
        elif lag > 0:
            x = np.zeros([Tb - 1, lag])
            for i in range(lag, Tb - 1):
                # create a new empy array of simlar proportions to dyb
                x = np.zeros([Tb - 1, lag])
                for k in range(lag):
                    # every row after the first 6, fill the first six column values with
                    # values of the dyb six rows that came before it
                    x[i, k] = dyb[i - k, j]

                # matrix multiplication
                # fill the rows below the first 6 with
                # the i row of x *
                dyb[i, j] = np.dot(x[i,], beta[1:g]) + wn[i - lag, j] * eps[rN[i - lag, j]]

    dyb0 = np.ones([Tb, nboot]) * y[1]
    dyb0[1:, :] = dyb
    yb = np.cumsum(dyb0, axis=0)

    dim = Tb - swindow0 + 1
    i = 0

    # for every .. column perform PSY, since there are 99 columns...
    # this gives a new matrix with 24 columns and 4 rows... so every row= ser
    MPSY = []
    for col in range(nboot):
        MPSY.append(PSY(pd.Series(yb[:, col]), swindow0, IC, adflag))

    MPSY = np.array(MPSY)
    # then, find the max value for each point in time?
    SPSY = MPSY.max(axis=1)
    # then, find the quantile for each
    Q_SPSY = pd.Series(SPSY.T[0]).quantile(qe)

    return Q_SPSY


def is_end_date(value, next_value):
    """determine if this is the end date of a time series"""
    if value != next_value - 1:
        return True
    else:
        return False


def is_start_date(value, previous_value):
    """determine if this is the start date of a time series."""
    if value != previous_value + 1:
        return True
    else:
        return False


def find_sequences_datetime(p, md):
    """
    Transform bubble occurence time sequences to a series of sequences
    :param p: list of periods with bubbles in date string format
    :param md: list of all dates of interest
    :return: Dataframe with start and end dates of a bubble.
    """
    all_dates = pd.to_datetime(md)

    locs = []
    for idx, date in enumerate(p):
        locs.append((all_dates == date).argmax())

    end_dates = [is_end_date(value, next_value) for value, next_value in zip(locs[:-1], locs[1:])] + [True]
    start_dates = [True] + [is_start_date(value, previous_value) for value, previous_value in zip(locs[1:], locs[:-1])]

    end_locs = np.array(locs)[np.array(end_dates)]
    start_locs = np.array(locs)[np.array(start_dates)]

    return pd.DataFrame({'end_date': all_dates[(end_locs)], 'start_date': all_dates[(start_locs)]})[
        ['start_date', 'end_date']]


def find_sequences_ints(p, md):
    """
    Transform bubble occurence time sequences of ints to a series of sequences
    :param p: list of periods with bubbles in date string format
    :param md: list of all dates of interest
    :return: Dataframe with start and end dates of a bubble.
    """
    locs = []
    for date in p:
        locs.append((md == date).argmax())

    end_dates = [is_end_date(value, next_value) for value, next_value in zip(locs[:-1], locs[1:])] + [True]
    start_dates = [True] + [is_start_date(value, previous_value) for value, previous_value in zip(locs[1:], locs[:-1])]

    end_locs = np.array(locs)[np.array(end_dates)]
    start_locs = np.array(locs)[np.array(start_dates)]

    return pd.DataFrame({'end_date': md[(end_locs)], 'start_date': md[(start_locs)]})[
        ['start_date', 'end_date']]


def bubble_period(all_dates, bubbly_date_serie):
    """
    Return all dates of a single bubble period given a start, end date and the full time series
    :param all_dates: list of all dates
    :param bubbly_date_serie: list of dates in which there was a bubble.
    :return:
    """
    first_date = (all_dates == bubbly_date_serie['start_date']).argmax()
    second_date = (all_dates == bubbly_date_serie['end_date']).argmax()

    if first_date == second_date:
        return all_dates[first_date-2:first_date]
    else:
        return all_dates[first_date:second_date + 1]


def p_bubbles(bubbly_dates):
    lenghts_of_bubbles = []
    for row in range(len(bubbly_dates)):
        lenghts_of_bubbles.append(bubbly_dates.iloc[row]['end_date'] - bubbly_dates.iloc[row]['start_date'] + 1)
    lenghts_of_bubbles = np.array(lenghts_of_bubbles)
    av_lenghts_of_bubbles = np.mean(lenghts_of_bubbles)
    long_bubble_condition = lenghts_of_bubbles > av_lenghts_of_bubbles
    r = np.array(range(len(long_bubble_condition)))
    locs_long_bubbles = r[long_bubble_condition]
    return locs_long_bubbles