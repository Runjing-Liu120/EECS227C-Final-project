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

N = 10     # observations per group
K = 2      # dimension of regressors
NG = 6      # number of groups



# Generate data
NObs = NG * N

# variances
prior_var = {'beta':10, 'mu':100, 'y':1}

beta = np.random.multivariate_normal(np.ones(K)*5, np.identity(K))
#beta = np.arange(K)
#beta = beta - np.mean(beta)

mu = np.random.normal(10, 20, NG)
prior_var = {'beta':100., 'mu':100., 'y':1.}

X = np.random.random(K * NObs).reshape(K, NObs) - 0.5
#x_mat = np.random.random(K * NObs).reshape(NObs, K) - 0.5
#X = np.identity(1)

# try with correlated design matrix


y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
y_mean = np.dot(X.T, beta) + mu[y_g_vec]

y_vec = np.random.normal(y_mean, prior_var['y'], NObs)



# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

iterations = 1000
beta_error = np.zeros(iterations+1)
mu_error = np.zeros(iterations+1)

beta_error[0] = np.linalg.norm(beta_mean - beta)
mu_error[0] = np.linalg.norm(mu_mean - mu)

for i in range(iterations):
    [beta_mean, mu_mean] = lmm_CAVI(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)
    #print beta_mean, mu_mean
    #print mu
    beta_error[i+1] = np.linalg.norm(beta_mean - beta)
    mu_error[i+1] = np.linalg.norm(mu_mean - mu)
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



# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

results, _, elbo = lmm_Newton(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)

plt.plot(elbo)
plt.show()

beta_post_mean = results.x[:K]
mu_post_mean   = results.x[K:]

print(beta_post_mean,mu_post_mean)
print(beta,mu)


from LMM_lib import KLWrapper

par1 = np.concatenate((beta_post_mean, mu_post_mean))
par2 = np.concatenate((beta, mu))
kl_wrapper = KLWrapper(par1, X, y_vec, beta_var, mu_var, prior_var, NG,N,K)


print "\n\n"
print kl_wrapper.kl(par1, X, y_vec, beta_var, mu_var, prior_var, NG,N,K)
print kl_wrapper.kl(par2, X, y_vec, beta_var, mu_var, prior_var, NG,N,K)