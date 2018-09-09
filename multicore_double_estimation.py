from pandas_datareader import data
import pandas as pd
import random
from SALib.sample import latin
from functions.stylizedfacts import *
import scipy.stats as stats
from functions.evolutionaryalgo import *
from pandas_datareader import data
from functions.helpers import hurst, organise_data, div_by_hundred, discounted_value_cash_flow, find_horizon, calculate_npv
import matplotlib.pyplot as plt
import quandl
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import json

empirical_moments = np.array([ -8.77599993e-03,  -9.83949423e-02,  -5.64810021e-02,
         3.39868973e-01,   1.23281435e+01,   3.52022421e-01,
         2.73786709e-01,   1.99870778e-01,   1.87612540e-01,
        -3.39594806e+00])

