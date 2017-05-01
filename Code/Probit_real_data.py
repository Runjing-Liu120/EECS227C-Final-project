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
from time import time

# import data
data = np.loadtxt("../Data/spambase.data", delimiter=',')
X = data[:,0:-1]
X = X.T # feature vectors in columns

N = np.shape(X)[1]

X = np.concatenate((np.ones((1,N)),X))

D = np.shape(X)[0]


print('size of data, (D,N): ', np.shape(X))

t = 2*data[:,-1] - 1 ## WAS (0,1) valued, WANT (-1,1) valued

v_0 = 100 # prior variance of w

np.random.seed(12345676)

w_init = np.random.multivariate_normal(np.zeros(D), np.identity(D))
z_init = np.random.normal(0, 1, N)

"""
# Gibbs sampler
#Gibbs_iterations = 10**5
#burn = 10**2
#[w_post_mean, z_post_mean] = probit_Gibbs(X, t, v_0, burn, Gibbs_iterations)

#print('Gibbs posterior mean: \n', w_post_mean)

print(method, ' posterior mean: \n', w_mean)
# print(z_loc)

"""

"""


## Newton method

# initialize
w_mean = deepcopy(w_init)
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))
z_loc = deepcopy(z_init)

results, times, elbo, pars = probit_Newton(X, t, v_0, w_mean, w_var, z_loc)
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
output0 = open('outputs/Newton_times_probit.pickle', 'wb')
pickle.dump(times, output0)

output1 = open('outputs/Newton_elbo_probit.pickle', 'wb')
pickle.dump(elbo, output1)

output2 = open('outputs/Newton_Wmean_probit.pickle', 'wb')
pickle.dump(w_post_mean, output2)

output3 = open('outputs/Newton_pars_probit.pickle', 'wb')
pickle.dump(pars, output3)

"""



# re-initialize
w_mean = deepcopy(w_init)
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))

z_loc = deepcopy(z_init)
z_var = 1


#output1 = open('outputs/Wvar_probit.pickle', 'wb')
#pickle.dump(w_var, output1)

#output2 = open('outputs/Newton_Wmean_probit.pickle', 'rb')
#w_post_mean = pickle.load(output2)



iterations = 3000
delta = np.zeros(iterations)
elbo  = np.zeros(iterations)
times = np.zeros(iterations)
#error = np.zeros(iterations+1)
#error[0] = np.linalg.norm(w_mean - w_post_mean)
pars = []

method = 'CAVI' # CAVI or PXVB

telbo = 0.
delbo = 0.
t0 = time()
i=0
delta[0] = 1.
#while i<iterations and delta[i] > 1e-6:
for i in range(iterations):
    #times[i] = time() - t0 - delbo
    w_mean_prev = deepcopy(w_mean)

    #telbo = time()
    # compute elbo
    par = np.concatenate((w_mean, z_loc))
    pars.append(par)
    elbo[i] = get_elbo(par, X, t, v_0, w_var)
    #telbo = time() - telbo 
    #delbo += telbo

    # CAVI updates
    if method == 'CAVI': 
        [w_mean, w_var, z_loc, z_trunc_mean] \
            = probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var)
    
    # PX-VB updates
    if method == 'PXVB': 
        [w_mean, w_var, z_loc, z_trunc_mean] \
            = probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var)
            
        [w_mean, w_var] = \
            probit_reparam(X, t, v_0, w_mean, w_var, z_loc, z_var)
    
    delta[i] = np.linalg.norm(w_mean - w_mean_prev)
    #error[i+1] = np.linalg.norm(w_mean - w_post_mean)

    
    if (i % 10) == 0:
        print(i)

# save output of method
output3 = open('outputs/' + method + '_elbo_probit_int.pickle', 'wb')
pickle.dump(elbo, output3)

output4 = open('outputs/' + method + '_Wmean_probit_int.pickle', 'wb')
pickle.dump(w_mean, output4)

output5 = open('outputs/' + method + '_times_probit_int.pickle', 'wb')
pickle.dump(times, output5)

output6 = open('outputs/' + method + '_pars_probit_int.pickle', 'wb')
pickle.dump(pars, output6)
