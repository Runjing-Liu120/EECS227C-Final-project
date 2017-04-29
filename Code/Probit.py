# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:03:24 2017

@author: Haiying Liang
"""
from Probit_lib import *
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib.pyplot as plt


# generate data

# np.random.seed(21312)
np.random.seed(314)

N = 10
D = 5

v_0 = 100 # prior variance of w

X = np.random.normal(0, 1, (D,N)) # design matrix

w = np.random.multivariate_normal(np.zeros(D), np.identity(D))

z = np.random.multivariate_normal(np.dot(w,X), np.identity(N))

t = np.sign(z)

# initializations
w_mean = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
#w_mean = deepcopy(w)
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))

z_loc = np.random.normal(0, 1, N)
#   z_loc = deepcopy(z)
z_var = 1



iterations = 100
error = np.zeros(iterations)
delta = np.zeros(iterations)

#method = 'PX-VB'
method = 'CAVI'

for i in range(iterations):
    w_mean_prev = deepcopy(w_mean)
    
    
    # CAVI updates
    if method == 'CAVI': 
        [w_mean, w_var, z_loc, z_trunc_mean] \
            = probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var)
        
    # PX-VB updates
    if method == 'PX-VB': 
        [w_mean, w_var, z_loc, z_trunc_mean] \
            = probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var)
            
        [w_mean, w_var] = \
            probit_reparam(X, t, v_0, w_mean, w_var, z_loc, z_var)
        
    error[i] = np.linalg.norm(w_mean - w)
    delta[i] = np.linalg.norm(w_mean - w_mean_prev)
    
    

plt.figure(1)
#plt.clf()
plt.semilogy(error)
plt.show()

plt.figure(2)
#plt.clf()
plt.semilogy(delta)


"""
# Gibbs sampler
#Gibbs_iterations = 10**5
#burn = 10**2
#[w_post_mean, z_post_mean] = probit_Gibbs(X, t, v_0, burn, Gibbs_iterations)

#print('Gibbs posterior mean: \n', w_post_mean)

print(method, ' posterior mean: \n', w_mean)
# print(z_loc)
#print(method, ' posterior mean: \n', w_mean)
#print(z_loc)

"""
"""
# re-initialize
w_mean = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
w_var  = np.identity(D)
z_loc = np.random.normal(0, 1, N)

results, times, elbo = probit_Newton(X, t, v_0, w_mean, w_var, z_loc)
w_post_mean = results.x[:D]
z_post_loc  = results.x[D:]

print("\n\n\n")
print(elbo)


plt.plot(elbo)
plt.xlabel("Time (seconds)")
plt.ylabel("ELBO")
plt.show()
"""