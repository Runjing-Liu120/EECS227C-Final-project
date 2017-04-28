# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:35:36 2017

@author: Haiying Liang
"""
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib.pyplot as plt

def lmm_CAVI(X, y, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K):

    # update mu parameters
    for g in range(NG): 
        #mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)
        mu_mean[g] = mu_var * (1/prior_var['mu']) * np.sum(y[i] - np.dot(X[:,i],beta_mean) for i in range(g*N, (g+1)*N))
        
                
                
    # update beta
    # beta_var = np.linalg.inv(1/prior['beta']*np.identity(K) \
        #                          + 1/prior_var['y'] * np.dot(X,X.T))
    
    y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
    canonical_beta_mean = 1/prior_var['y'] * np.dot(X, y - mu_mean[y_g_vec])
    beta_mean = np.dot(beta_var, canonical_beta_mean)
    
    return(beta_mean, mu_mean)

