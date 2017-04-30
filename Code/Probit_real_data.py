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
import pickle

# import data
data = np.loadtxt("../Data/spambase.data", delimiter=',')
X = data[:,0:-1]
X = X.T # feature vectors in columns

N = np.shape(X)[1]
D = np.shape(X)[0]

print('size of data, (D,N): ', np.shape(X))

t = data[:,-1]

v_0 = 100 # prior variance of w


"""
# Gibbs sampler
#Gibbs_iterations = 10**5
#burn = 10**2
#[w_post_mean, z_post_mean] = probit_Gibbs(X, t, v_0, burn, Gibbs_iterations)

#print('Gibbs posterior mean: \n', w_post_mean)

print(method, ' posterior mean: \n', w_mean)
# print(z_loc)

"""

## Newton method

# initialize
w_mean = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))
z_loc = np.random.normal(0, 1, N)

results, times, elbo = probit_Newton(X, t, v_0, w_mean, w_var, z_loc)
w_post_mean = results.x[:D]
z_post_loc  = results.x[D:]

print("\n\n\n")
print(elbo)


plt.figure(3)
plt.clf()
plt.plot(elbo)
plt.xlabel("Time (seconds)")
plt.ylabel("ELBO")
plt.show()

# save output of Newton
output1 = open('Newton_elbo_probit.pickle', 'wb')
pickle.dump(elbo, output1)

output2 = open('Newton_Wmean_probit.pickle', 'wb')
pickle.dump(w_post_mean, output2)

#***************
"""

# re-initializations
w_mean = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))

z_loc = np.random.normal(0, 1, N)
z_var = 1


iterations = 1000
delta = np.zeros(iterations)
elbo = np.zeros(iterations)
error = np.zeros(iterations+1)
error[0] = np.linalg.norm(w_mean - w_poast_mean)
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
    
    delta[i] = np.linalg.norm(w_mean - w_mean_prev)
    error[i+1] = np.linalg.norm(w_mean - w_post_mean)
    # compute elbo
    par = np.concatenate((w_mean, z_loc))
    elbo_CAVI[i] = get_elbo(par, X, t, v_0, w_var)
    
    if (i % 10) == 0:
        print(i)

    
plt.figure(1)
plt.clf()
plt.semilogy(delta)
plt.xlabel('iteration')
plt.ylabel('$\|w^{l+1} - w^l\|')
plt.title('pairwise difference')
plt.show()

plt.figure(2) 
plt.clf()
plt.plot(error)
plt.title('CAVI error to Newton')
plt.xlabel('iteration')

plt.figure(3)
plt.plot(elbo_CAVI)

# save output of CAVI
output3 = open('CAVI_elbo_probit.pickle', 'wb')
pickle.dump(elbo_CAVI, output3)

output4 = open('CAVI_Wmean_probit.pickle', 'wb')
pickle.dump(w_mean, output4)
"""