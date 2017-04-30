# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:54:06 2017

@author: Haiying Liang

Probit testing
"""

from Probit_lib import *
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib.pyplot as plt
import sys


# test the means of trucated normal
z_loc = np.array([10.])

samples = np.random.normal(z_loc, 1., 10**7)

samples_trunc = samples[samples<0]

[z_trunc_mean, z_trunc_var] = trunc_Normal(z_loc, -1.)
print(np.shape(samples_trunc))

print('sampled mean: ', np.average(samples_trunc))
print('computed mean: ', z_trunc_mean)
print('\nsampled variance: ', np.var(samples_trunc))
print('computed variance: ', z_trunc_var)


from scipy.stats import truncnorm

mean, var, skew, kurt = truncnorm.stats(a=-1e6, b=-z_loc, loc=z_loc, scale=1., moments='mvsk')
print mean, var