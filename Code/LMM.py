# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:36:21 2017

@author: Haiying Liang
"""
from LMM_lib import *
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib.pyplot as plt

N = 200     # observations per group
K = 5      # dimension of regressors
NG = 200      # number of groups



# Generate data
NObs = NG * N

# variances
prior_var = {'beta':100, 'mu':100, 'y':1}

beta = np.random.multivariate_normal(np.zeros(K), np.identity(K))

mu = np.random.normal(0, 1, NG)

X = np.random.random(K * NObs).reshape(K, NObs) - 0.5
y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
y_mean = np.dot(X.T, beta) + mu[y_g_vec]

y_vec = np.random.normal(y_mean, prior_var['y'], NObs)

# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
#beta_mean = deepcopy(beta)
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

iterations = 20
beta_error = np.zeros(iterations+1)
mu_error = np.zeros(iterations+1)

beta_error[0] = np.linalg.norm(beta_mean - beta)
mu_error[0] = np.linalg.norm(mu_mean - mu)

for i in range(iterations):
    [beta_mean, mu_mean] = lmm_CAVI(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)
    beta_error[i+1] = np.linalg.norm(beta_mean - beta)
    mu_error[i+1] = np.linalg.norm(mu_mean - mu)
    
    
plt.figure(1)
plt.clf()
plt.plot(beta_error)
plt.title('beta error')
    
plt.figure(2)
plt.clf()
plt.plot(mu_error)
plt.title('mu error')


