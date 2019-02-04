import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller


def ADF(y, IC=0, adflag=0):
    """
    Calculates the augmented Dickey-Fuller (ADF) test statistic with lag order set fixed or selected by AIC or BIC.

    Port from: https://github.com/itamarcaspi/psymonitor/
    Credits to: Phillips, P. C. B., Shi, S., & Yu, J. (2015a).
    Testing for multiple bubbles: Historical episodes of exuberance and collapse in the S&P 500.
    International Economic Review, 56(4), 1034–1078.

    :param y: list data
    :param IC:
    :param adflag:
    :return: float ADF test statistic.
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
                ICC[k,] = -2 * npdf / float(t) + 2 * len(beta) / float(t)  # TODO check if correct
            elif IC == 2:
                ICC[k,] = -2 * npdf / float(t) + len(beta) * np.log(t) / float(t)
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
    bsadf = bsadfs[swindow0-1 : t] #TODO check if this is correct

    return bsadf


def cvPSYwmboot(y, swindow0, adflag, control_sample_size, nboot=199, nCores=1):  # IC = 'BIC'
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
    confidence_levels =  np.array([0.90, 0.95, 0.99])

    result = adfuller(y, maxlag=adflag, regression="c", autolag='BIC', store=True, regresults=True)[-1].resols # list of ADF [beta, errors, lag]
    beta = result.params
    eps = result.resid
    lag = adflag

    T0 = len(eps)
    t = len(y)
    dy = np.array(y.iloc[0:(t - 1)]) - np.array(y.iloc[1:t]) #difference of the data
    g = len(beta)

    if not swindow0:
        swindow0 = math.floor(t * (0.01 + 1.8 / np.sqrt(t)))

    # The Data generating process (DGP)
    np.random.seed(101)
    # create matrix filled with random ints < T0 with rows TB and cols: nboot
    random_numbers = np.random.randint(0, T0, (control_sample_size, nboot))
    # create matrix filled with random normal floats with rows TB and cols: nboot
    random_normal_numbers = np.random.normal(1, size=(control_sample_size, nboot))

    dyb = np.zeros([control_sample_size - 1, nboot])
    dyb[:lag, ] = np.split(np.tile(dy[:lag], nboot), lag, axis=0) # make the first six rows equal to a repeat of the differences of that lag

    for j in range(nboot):
        # loop over all columns
        if lag == 0:
            for i in range(lag + 1, control_sample_size - 1):
                # loop over rows, start filling the rest of the rowswith random numbers
                dyb[i, j] = random_normal_numbers[i - lag, j] * eps[random_numbers[i - lag, j]]

        elif lag > 0:
            #x = np.zeros([control_sample_size - 1, lag])
            for i in range(lag + 1, control_sample_size - 1):
                x = np.zeros([control_sample_size - 1, lag])
                for k in range(lag):
                    x[i, k] = dyb[(i - k), j]

                # matrix multiplication
                dyb[i, j] = np.dot(x[i, ], beta[2:g, 1]) + random_normal_numbers[i - lag, j] * eps[random_numbers[i - lag, j]]


    yb0 = np.repeat(y[1], repeats=nboot)
    dyb0 = np.concatenate(yb0, dyb)
    yb = apply(dyb0, 2, np.cumsum)

    # The PSY Test ------------------------------------------------------------

    # setup parallel backend to use many processors
    if nCores > 1:
        nCores = nCores
    else:
        nCores = 1

    #cl = makeCluster(nCores)
    #registerDoParallel(cl)

    # # ----------------------------------
    # dim = Tb - swindow0 + 1
    # i = 0
    #
    # MPSY = foreach(i=1:nboot, .inorder = FALSE, .combine = rbind) % dopar %:
    #     PSY(yb[, i], swindow0, IC, adflag)
    #
    # SPSY = apply(MPSY, 1, max()) # apply MPSY function
    # Q_SPSY = np.quantile(SPSY, confidence_levels)

    #return Q_SPSY
    pass


