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

# 10, 2, 6
N = 1     # observations per group
K = 10      # dimension of regressors
NG = 10      # number of groups

##### GENERATE DATA #####
NObs = NG * N

# variances
prior_var = {'beta':float('Inf'), 'mu':100., 'y':1.}

beta = np.random.normal(0,1,K)
mu = np.random.normal(0, 1, NG)

#X = np.random.random(K * NObs).reshape(K, NObs) - 0.5
X = np.identity(NG)

y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
y_mean = np.dot(X.T, beta) + mu[y_g_vec]

y_vec = np.random.normal(y_mean, prior_var['y'], NObs)


"""

##### NEWTON METHOD #####
print("Running Newton")

# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))

mu_mean = np.random.normal(0,1, NG)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

results, _, elbo_newt, pars = lmm_Newton(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K,maxiter=10)
delta_newt_beta = [np.linalg.norm(pars[i][:K] - pars[i+1][:K]) for i in range(len(pars)-1)] 
delta_newt_mu   = [np.linalg.norm(pars[i][K:] - pars[i+1][K:]) for i in range(len(pars)-1)] 
d_newt_beta = [np.linalg.norm(par[:K] - beta) for par in pars]
d_newt_mu   = [np.linalg.norm(par[K:] - mu) for par in pars]

beta_post_mean = results.x[:K]
mu_post_mean   = results.x[K:]
"""

##### CAVI METHOD #####
print("Running CAVI")

# variational parameters
beta_mean = np.random.multivariate_normal(np.zeros(K), np.identity(K))
beta_mean_p = deepcopy(beta_mean)

beta_var = np.linalg.inv(1/prior_var['beta']*np.identity(K) \
                                  + 1/prior_var['y'] * np.dot(X,X.T))
#beta_var = np.linalg.inv(1/prior_var['y'] * np.dot(X,X.T))


mu_mean = np.random.normal(0,1, NG)
mu_mean_p = deepcopy(mu_mean)
mu_var = (1/prior_var['mu'] + N/prior_var['y'])**(-1)

iterations = 5000
d_cavi_beta = np.zeros(iterations)
d_cavi_mu   = np.zeros(iterations)
d_pxvb_beta = np.zeros(iterations)
d_pxvb_mu   = np.zeros(iterations)

elbo_cavi       = np.zeros(iterations)

for i in range(iterations):
	beta_mean_prev = deepcopy(beta_mean)
	mu_mean_prev   = deepcopy(mu_mean)

	elbo_cavi[i] = get_elbo(np.concatenate((beta_mean, mu_mean)), X, y_vec, beta_var, mu_var, prior_var, NG,N,K)

	d_pxvb_beta[i] = np.linalg.norm(beta_mean_p - y_vec)
	d_pxvb_mu[i] = np.linalg.norm(mu_mean_p - 0)
	d_cavi_beta[i] = np.linalg.norm(beta_mean - y_vec)
	d_cavi_mu[i]   = np.linalg.norm(mu_mean - 0)
     
	[beta_mean, mu_mean] = lmm_CAVI(X, y_vec, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K)
      
      # PXVB
	[beta_mean_p, mu_mean_p] = lmm_CAVI(X, y_vec, beta_mean_p, beta_var, mu_mean_p, mu_var, prior_var, NG,N,K)
	mu_mean_p = mu_mean_p - mu_mean_p
 

print(np.linalg.norm(beta_mean - beta))
print(np.linalg.norm(mu_mean - mu))
print(np.linalg.norm(beta_post_mean - beta))
print(np.linalg.norm(mu_post_mean - mu))

"""
plt.figure(1)
plt.clf()
plt.plot(beta_error)
plt.title('beta error')


plt.figure(2)
plt.clf()
plt.plot(mu_error)
plt.title('mu error')
plt.show()
"""


plt.figure(1)
plt.clf()
plt.plot(elbo_cavi[1:],'r')
# plt.plot(elbo_newt[1:],'g')
plt.xlabel("iter") 
plt.ylabel("ELBO")
plt.legend(['CAVI','NCG'])


plt.figure(2)
plt.clf()
plt.semilogy(d_cavi_mu,'r')
plt.semilogy(d_pxvb_mu,'B')
#plt.plot(d_newt_mu,'b')
plt.xlabel("iter") 
plt.title("Distance to minimizer (mu)")
plt.legend(['CAVI','PXVB'])

plt.figure(3)
plt.clf()
plt.semilogy(d_cavi_beta,'r')
plt.semilogy(d_pxvb_beta,'b')
#plt.plot(d_newt_beta,'b')
plt.xlabel("iter") 
plt.title("Distance to minimizer (beta)")
plt.legend(['CAVI','PXVB'])
"""
plt.figure(4)
plt.plot(delta_cavi_mu,'r')
plt.plot(delta_newt_mu,'b')
plt.xlabel("iter") 
plt.title("Pairwise Distances")
plt.legend(['CAVI','NCG'])

plt.figure(5)
plt.plot(delta_cavi_beta,'r')
plt.plot(delta_newt_beta,'b')
plt.xlabel("iter")
plt.title("Pairwise Distances")
plt.legend(['CAVI','NCG'])
"""
plt.show()