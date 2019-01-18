import numpy as np
import matplotlib.pyplot as plt


def gini(array):
    """
    Calculate the Gini coefficient of a numpy array.
    All credits to Olivia Guest @ https://github.com/oliviaguest/gini
    based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    array = array.flatten()

    # make sure values are not negative
    if np.amin(array) < 0:
        array -= np.amin(array)
        print('Negative values founds, check calculation')

    # slightly offset values of 0
    array += 0.0000001

    # sort values
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0] # number of array elements

    gini_coefficient = ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    return gini_coefficient


def lorenz_curve(array):
    """
    Plot lorenz curve and reference equality line plot
    :param array: sorted array for which you want to plot the lorenz curve
    :return:
    """
    lorenz = array.cumsum() / array.sum()
    lorenz = np.insert(lorenz, 0, 0)

    fig, ax = plt.subplots(figsize=[6,6])
    # scatter plot of Lorenz curve
    ax.plot(np.arange(lorenz.size)/float((lorenz.size-1)), lorenz)
    # line plot of equality
    ax.plot([0,1], [0,1], color='k')