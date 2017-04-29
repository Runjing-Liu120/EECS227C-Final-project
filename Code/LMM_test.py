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

np.random.seed(123242)

# Simulate data
N = 10     # observations per group
K = 5      # dimension of regressors
NG = 30      # number of groups

# Generate data
NObs = NG * N
true_beta = np.array(range(K))
true_beta = true_beta - np.mean(true_beta)
true_y_sd = 1.0
true_y_info = 1 / true_y_sd**2

true_mu = 1.0
true_mu_sd = 20.0
true_mu_info = 1 / true_mu_sd**2
true_u_sufficient = np.random.normal(true_mu, 1 / np.sqrt(true_mu_info), NG)
true_u_ancillary = true_u_sufficient - true_mu

x_mat = np.random.random(K * NObs).reshape(NObs, K) - 0.5
x_rot = np.full((K, K), 0.5)
for k in range(K):
    x_rot[k, k] = 1.0
X = np.matmul(x_mat, x_rot).T



y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
true_mean = np.matmul(x_mat, true_beta) + true_u_sufficient[y_g_vec]
y_vec = np.random.normal(true_mean, 1 / np.sqrt(true_y_info), NG * N)


prior_var = {'mu': true_mu_sd**2, 'y': true_y_sd**2, 'beta': 100}


#print X.shape, y_vec.shape, 

##### NEWTON METHOD #####

# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

results, _, elbo = lmm_Newton(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)

#plt.plot(elbo)
#plt.show()

beta_post_mean = results.x[:K]
mu_post_mean   = results.x[K:]

print(beta_post_mean,mu_post_mean)

##### NEWTON METHOD #####







# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

iterations = 10
beta_error = np.zeros(iterations+1)
mu_error = np.zeros(iterations+1)

#beta_error[0] = np.linalg.norm(beta_mean - beta)
#mu_error[0] = np.linalg.norm(mu_mean - mu)

for i in range(iterations):
    [beta_mean, mu_mean] = lmm_CAVI(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)
    #print beta_mean, mu_mean
    #print mu
    #beta_error[i+1] = np.linalg.norm(beta_mean - beta)
    #mu_error[i+1] = np.linalg.norm(mu_mean - mu)
    #print(beta_mean)

print(beta_mean,mu_mean)

    
"""
plt.figure(1)
plt.clf()
plt.plot(beta_error)
plt.title('beta error')
plt.show()

plt.figure(2)
plt.clf()
plt.plot(mu_error)
plt.title('mu error')
plt.show()


"""