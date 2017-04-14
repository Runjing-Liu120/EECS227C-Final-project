# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:18:08 2017

@author: Haiying Liang
"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Create data
V = 5

theta = 10 # totally a random draw from the reals
mu = np.random.normal(theta, V)
Y = np.random.normal(mu,1)
    

# variational parameters
VI_param = {'mu_mean': 0, 'mu_var': 1, \
            'theta_mean': 0, 'theta_var': 1,\
            'nu_mean': 0, 'nu_var': 1}

            
# updates in the sufficient model
def sufficient_CAVI(iterations, VI_param, V, Y):
    mu_mean = VI_param['mu_mean']
    theta_mean = VI_param['theta_mean'] 
    mu_var = VI_param['mu_var']    
    theta_var = VI_param['theta_var']
    
    elbo = np.zeros(iterations)
    theta_mean_stored = np.zeros(iterations)
    
    for i in range(iterations):
        mu_mean = (Y*V + theta_mean) / (1 + V)

        theta_mean = mu_mean

        mu_var = V/(1+V)
        
        theta_var = V

        # elbo calculation
        # expectation of log P(Y|mu) 
        Expectation1 = - 0.5*(-2*Y*mu_mean + mu_mean**2 + mu_var) 
        # epxectation of log P(mu|theta)
        Expectation2 = - 1/(2*V) * (mu_mean**2 + mu_var - \
                            2*mu_mean*theta_mean + theta_mean**2 + theta_var)
        #entropy terms
        Entropy = 0.5*(np.log(mu_var) + np.log(theta_var))
        
        theta_mean_stored[i] = theta_mean
        elbo[i] = Expectation1 + Expectation2 + Entropy
    
    plt.figure(1)
    plt.clf()
    plt.plot(elbo)
    plt.xlabel('iteration')
    plt.ylabel('elbo')
    
    plt.figure(2)
    plt.clf()
    plt.semilogy(abs(theta_mean_stored - Y))
    plt.xlabel('iteration')
    plt.ylabel('$|E_q(theta) - Y|$')
    
    VI_param['mu_mean'] = mu_mean
    VI_param['theta_mean'] = theta_mean
    VI_param['mu_var'] =   mu_var
    VI_param['theta_var'] = theta_var

    return(VI_param)

iterations = 5
sufficient_CAVI(iterations, VI_param, V, Y)

def ancillary_CAVI(iterations, VI_param, V, Y):
    nu_mean = VI_param['nu_mean']
    theta_mean = VI_param['theta_mean'] 
    nu_var = VI_param['nu_var']    
    theta_var = VI_param['theta_var']
    
    elbo = np.zeros(iterations)
    theta_mean_stored = np.zeros(iterations)
    
    for i in range(iterations):
        nu_mean = V*(Y - theta_mean) / (1 + V)

        theta_mean = Y - nu_mean

        nu_var = V/(1+V)
        
        theta_var = 1

        # elbo calculation
        # expectation of log P(Y|mu) 
        Expectation1 = - 0.5*(-2*Y*(nu_mean + theta_mean)\
                          + nu_mean**2 + 2*nu_mean*theta_mean + theta_mean**2\
                          + nu_var + theta_var) 
        Expectation1 = 0.5*(2*Y*nu_mean + 2*Y*theta_mean - 2*nu_mean*theta_mean - nu_var - nu_mean**2 - theta_var - theta_mean**2)
        
        # epxectation of log P(mu|theta)
        Expectation2 = - 1/(2*V) * (nu_mean**2 + nu_var)
                            
        #entropy terms
        Entropy = 0.5 * (np.log(nu_var) + np.log(theta_var))
        
        theta_mean_stored[i] = theta_mean
        elbo[i] = Expectation1 + Expectation2 + Entropy
    
    plt.figure(1)
    plt.clf()
    plt.plot(elbo)
    plt.xlabel('iteration')
    plt.ylabel('elbo')
    
    plt.figure(2)
    plt.clf()
    plt.plot(abs(theta_mean_stored - Y))
    plt.xlabel('iteration')
    plt.ylabel('$|E_q(theta) - Y|$')
    
    VI_param['nu_mean'] = nu_mean
    VI_param['theta_mean'] = theta_mean
    VI_param['nu_var'] =   nu_var
    VI_param['theta_var'] = theta_var

    return(VI_param)

iterations = 5
ancillary_CAVI(iterations, VI_param, V, Y)

