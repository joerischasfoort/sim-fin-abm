import numpy as np
import pandas as pd
import math
from statsmodels.tsa.stattools import adfuller


def PSY(y, swindow0, adflag):
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
            rwadft[r1] = float(
                adfuller(y.iloc[r1:r2], maxlag=adflag, autolag='BIC')[0])  # two tail 5% significant level

        # take max value an insert in bsadfs array
        bsadfs[r2 - 1] = max(rwadft.T[0])

    # create shortened version of array
    bsadf = bsadfs[swindow0:t]

    return bsadf


def cvPSYwmboot(y, swindow0, adflag, Tb, nboot=199, nCores=1):  # IC = 'BIC'
    """
    Computes a matrix of 90, 95 and 99 critical values which can be used to compare to the bsadf statistics.
    param: y: data array or pandas df
    param: swindow0: integer minimum window size
    param: adflag: An integer, lag order when IC=0; maximum number of lags when IC>0 (default = 0).
    param: Tb: Integer the simulated sample size
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
    dy = np.array(y[2:t] - y[1:(t - 1)])
    g = len(beta)

    if not swindow0:
        swindow0 = math.floor(t * (0.01 + 1.8 / np.sqrt(t)))

    # The Data generating process (DGP)
    np.random.seed(101)
    # create matrix filled with random ints < T0 with rows TB and cols: nboot
    rN = np.random.randint(0, T0, (Tb, nboot))
    # create matrix filled with random normal floats with rows TB and cols: nboot
    wn = np.random.normal(1, size=(Tb, nboot))

    dyb = np.zeros([Tb - 1, nboot])
    dyb[1:lag, ] = np.repeat(dy[1:lag], repeats=nboot)

    for j in range(nboot):
        if lag == 0:
            for i in range(lag + 1,Tb):
                dyb[i, j] = wn[i - lag, j] * eps[rN[i - lag, j]]

        elif lag > 0:
            x = np.zeros(Tb-1, lag)
            for i in range(lag + 1, Tb):
                x = np.zeros(Tb-1, lag)
                for k in range(lag):
                    x[i, k] = dyb[(i - k), j]

                # matrix multiplication
                dyb[i, j] = np.dot(x[i, ], beta[2:g, 1] + wn[i - lag, j] * eps[rN[i - lag, j]])


    yb0 = np.repeat(y[1], repeats=nboot)
    dyb0 = np.concatenate(yb0, dyb)
    yb = apply(dyb0, 2, np.cumsum)

    # The PSY Test ------------------------------------------------------------

    # setup parallel backend to use many processors
    if nCores > 1:
        nCores = nCores
    else:
        nCores = 1

    cl = makeCluster(nCores)
    registerDoParallel(cl)

    # ----------------------------------
    dim = Tb - swindow0 + 1
    i = 0

    MPSY = foreach(i=1:nboot, .inorder = FALSE, .combine = rbind) % dopar %:
        PSY(yb[, i], swindow0, IC, adflag)

    SPSY = as.matrix(apply(MPSY, 1, max()))
    Q_SPSY = as.matrix(quantile(SPSY, qe))

    return Q_SPSY


