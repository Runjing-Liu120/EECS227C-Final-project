# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:55:09 2017

@author: Haiying Liang

Gamma-exponential hierchical model
"""

# generate data
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

np.random.seed(11)

# hyper parameters
gammas = [9,2]
beta = np.random.gamma(gammas[0], 1/gammas[1]) # draw beta

# draw mu given beta
alpha = 3 # fixed for simplicity
mu = np.random.gamma(alpha, beta)

# draw observation
Y = np.random.exponential(1/mu)

# run Gibbs sampler to compute posterior mean
Gibbs_iterations = 10**6
burn = 10*2
mu_sample = np.zeros(Gibbs_iterations)
beta_sample = np.zeros(Gibbs_iterations)

for i in range(Gibbs_iterations):
        mu_sample[i] = np.random.gamma(1+alpha, 1/(beta_sample[i-1]+Y))
        beta_sample[i] = np.random.gamma(alpha + gammas[0], \
                                    1/(mu_sample[i] + gammas[1]))
    
    
print('sufficient Gibbs sampler posterior means')
print('mu: ', np.average(mu_sample[burn:]))
print('beta: ', np.average(beta_sample[burn:]))


# Variational updates in the sufficient augmentation
def sufficient_CAVI(iterations, Variational_params, alpha, gammas, Y):
    # fixed gamma hyperparameters
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    
    # variational parameters 
    mu1 = Variational_params[0] # shape
    mu2 = Variational_params[1] # rate
    
    beta1 = Variational_params[2] # shape
    beta2 = Variational_params[3] # rate

    for i in range(iterations):
    # mu updates
        mu1 = alpha + 1
        mu2 = Y + beta1/beta2
        
        beta1 = alpha + gamma1 
        beta2 = mu1/mu2 + gamma2
    
    Variational_params = [mu1, mu2, beta1, beta2]
    
    
    return(Variational_params)

CAVI_iterations = 1000
Variational_params = np.random.uniform(0,10,4)
post_params = sufficient_CAVI(CAVI_iterations, \
                                     Variational_params, alpha, gammas, Y)

print('posterior means sufficient CAVI')
print('mu: ', post_params[0]/post_params[1] )
print('beta: ', post_params[2]/post_params[3] )




# run Gibbs sampler in ancillary model to compute posterior mean
Gibbs_iterations = 10**6
burn = 10*2
nu_sample = np.zeros(Gibbs_iterations)
beta_sample = np.zeros(Gibbs_iterations)

for i in range(Gibbs_iterations):
    nu_sample[i] = np.random.gamma(1+alpha, 1/(1+beta_sample[i-1]*Y))
    beta_sample[i] = np.random.gamma(1 + gammas[0], \
                                    1/(nu_sample[i]*Y + gammas[1]))
    
    
print('ancillary Gibbs sampler posterior means')
print('nu: ', np.average(nu_sample[burn:]))
print('beta: ', np.average(beta_sample[burn:]))
print('mu = nu*beta: ', np.average(nu_sample[burn:])*\
          np.average(beta_sample[burn:]))


# Variational updates in the ancillary model 
def ancillary_CAVI(iterations, Variational_params, alpha, gammas, Y):
    # fixed gamma hyperparameters
    gamma1 = gammas[0]
    gamma2 = gammas[1]
    
    # variational parameters 
    nu1 = Variational_params[0] # shape
    nu2 = Variational_params[1] # rate
    
    beta1 = Variational_params[2] # shape
    beta2 = Variational_params[3] # rate

    for i in range(iterations):
    # mu updates
        nu1 = alpha + 1
        nu2 = 1 + beta1/beta2*Y
        
        beta1 = 1 + gamma1 
        beta2 = nu1/nu2*Y + gamma2
                
    Variational_params = [nu1, nu2, beta1, beta2]
    
    
    
    return(Variational_params)

CAVI_iterations = 1000
Variational_params = np.random.uniform(0,10,4)
post_params = ancillary_CAVI(CAVI_iterations, \
                                     Variational_params, alpha, gammas, Y)

print('posterior means ancillary CAVI')
print('nu: ', post_params[0]/post_params[1] )
print('beta: ', post_params[2]/post_params[3] )
print('mu = beta*nu: ', post_params[0]/post_params[1] * post_params[2]/post_params[3] )
