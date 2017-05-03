# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:03:24 2017

@author: Haiying Liang
"""
from Probit_lib import *
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from time import time


# generate data

# np.random.seed(21312)
np.random.seed(12345676) # 314


N = 500
D = 20 # 10
v_0 = 10. # prior variance of w (was 1000.)


my_init_w = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
my_init_z = np.random.normal(0, 1, N)



#x_rot = np.eye(D) + np.full((D,D),0.5)
#col = np.random.normal(0, 1, (D))
#X = np.zeros((D,N))
#for n in range(N):
#    X[:,n] = col + np.random.normal(0, .0001, (D))
X = np.random.normal(0, 1., (D,N))
#np.dot(x_rot, np.random.normal(0, 1, (D,N))) # design matrix


w = np.random.multivariate_normal(np.zeros(D), np.identity(D))
z = np.random.multivariate_normal(np.dot(w,X), np.identity(N))
t = np.sign(z)

# NEWTON method
w_mean = deepcopy(my_init_w) #np.random.multivariate_normal(np.zeros(D), np.identity(D) )
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))
z_loc = deepcopy(my_init_z)#np.random.normal(0, 1, N)

results, times_newt, elbo_newt, pars = probit_Newton(X, t, v_0, w_mean, w_var, z_loc)
w_post_mean = results.x[:D]
z_post_loc  = results.x[D:]



# errors = [np.linalg.norm(par[:D] - w) for par in pars] 
# delta_newt = [np.linalg.norm(pars[i][:D] - pars[i+1][:D]) for i in range(len(pars)-1)] 
# d_newt = [np.linalg.norm(par[:D] - w) for par in pars] # wpostmean here 
#plt.figure(1)
#plt.semilogy(times, elbo,'b')

#w_post_mean = np.array([-228.2672737, 85.18464915, -38.21913674, -90.31504869, 165.28645233,
#                        -109.21586268, 80.58717793, -118.28729604, -223.92790542,  102.93045296])


# CAVI method
w_mean = deepcopy(my_init_w)  #deepcopy(w_post_mean) #np.random.multivariate_normal(np.zeros(D), np.identity(D) ) 
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))

z_loc = deepcopy(my_init_z)#np.random.normal(0, 1, N) #np.random.normal(0, 1, N)
z_var = 1

elbo_cavi = []
times_cavi = []
t0 = time()
PX_error2newt = []

i = 1
iterations = 100
CAVI_error2newt = []
delta_cavi = [1.]#np.zeros((iterations,1))
d_cavi = []
while delta_cavi[-1] > 1e-13: 
#for i in range(iterations):
    w_mean_prev = deepcopy(w_mean)
    d_cavi.append(np.linalg.norm(w - w_mean))
    
    # CAVI updates
    times_cavi.append(time() - t0)
    elbo_cavi.append(get_elbo(np.concatenate((w_mean,z_loc)), X, t, v_0, w_var))

    [w_mean, w_var, z_loc, z_trunc_mean] \
        = probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var)  

    delta_cavi.append(np.linalg.norm(w_mean_prev - w_mean))

    i+=1
    if i%100==0:
        # get_elbo(np.concatenate((w_mean,z_loc)), X, t, v_0, w_var,flag=True)
        print(i)
        print(delta_cavi[-1])
        
    CAVI_error2newt.append(np.linalg.norm(w_mean - w_post_mean))
    
delta_cavi = delta_cavi[1:]


# PX-VB method
w_mean_p = deepcopy(my_init_w) #np.random.multivariate_normal(np.zeros(D), np.identity(D) ) # d
w_var  = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))

z_loc_p = deepcopy(my_init_z)#np.random.normal(0, 1, N)
z_var = 1

elbo_pxvb = []
times_pxvb = []
t0 = time()

iterations = 100
PX_error2newt = []
delta_pxvb = [1.]
d_pxvb = []
i = 1
while delta_pxvb[-1] > 1e-13: 
    w_mean_prev = deepcopy(w_mean_p)
    
    # PX-VB updates
    times_pxvb.append(time() - t0)
    elbo_pxvb.append(get_elbo(np.concatenate((w_mean_p,z_loc_p)), X, t, v_0, w_var))
    [w_mean_p, w_var, z_loc_p, z_trunc_mean] \
        = probit_CAVI(X, t, v_0, w_mean_p, w_var, z_loc_p, z_var)
        
    [w_mean_p, w_var] = \
        probit_reparam(X, t, v_0, w_mean_p, w_var, z_loc_p, z_var)

    delta_pxvb.append(np.linalg.norm(w_mean_prev - w_mean_p))
    #d_pxvb.append(np.linalg.norm(w_post_mean - w_mean))
    i+=1
    if i%100==0:
        print(i)
    
    PX_error2newt.append(np.linalg.norm(w_mean_p - w_post_mean))

delta_pxvb = delta_pxvb[1:]


matplotlib.rcParams.update({'font.size': 14})

plt.figure(1)
plt.clf()
plt.plot(elbo_cavi[0:10],'r')
plt.plot(elbo_pxvb[0:10],'b')
plt.plot(elbo_newt[0:10],'g')
plt.xlabel("Iteration Number") 
plt.ylabel("ELBO")
plt.legend(['CAVI','PX-VB','NCG'], loc= 4)
plt.title('ELBO per Iteration')
plt.tight_layout()
plt.savefig('../poster/Probit_synth/elbo_iter.png')


plt.figure(2)
plt.clf()
plt.semilogy(times_cavi, delta_cavi,'r')
plt.semilogy(times_pxvb, delta_pxvb,'b') 
#plt.semilogy(delta_newt,'g')
plt.legend(['CAVI','PX-VB'])
plt.xlabel('Wall Time (s)')
plt.ylabel('$||w^{t+1} - w^t||_2$')
plt.title('Difference Between Consecutive Estimates')
plt.tight_layout()
plt.savefig('../poster/Probit_synth/CAVI_PX_convergence.png')


plt.figure(3)
plt.clf()
plt.plot(times_cavi[1:10], elbo_cavi[1:10],'ro-')
plt.plot(times_pxvb[1:10], elbo_pxvb[1:10],'bd-')
plt.plot(times_newt[1:4], elbo_newt[1:4],'g*-')
plt.xlabel("Wall Time (s)") 
plt.ylabel("ELBO")
plt.legend(['CAVI','PX-VB','NCG'], loc = 4)
plt.title('ELBO vs Time')
plt.tight_layout()
plt.savefig('../poster/Probit_synth/elbo_time.png')


plt.figure(4)
plt.clf()
plt.semilogy(times_cavi, CAVI_error2newt,'r')
plt.semilogy(times_pxvb, PX_error2newt,'b')
plt.xlabel("Wall Time (s)") 
plt.ylabel("Error $||w^t - w^*||_2$")
plt.legend(['CAVI','PX-VB'], loc = 4)
plt.title('Error to truth (as computed by NCG)')
plt.tight_layout()
plt.savefig('../poster/Probit_synth/error2newt_time.png')


plt.figure(5)
plt.clf()
plt.semilogy(CAVI_error2newt[0:100],'r')
plt.semilogy(PX_error2newt[0:100],'b')
plt.xlabel("Iteration Number") 
plt.ylabel("Error $||w^t - w^*||_2$")
plt.legend(['CAVI','PX-VB'], loc = 4)
plt.title('Error to truth (as computed by NCG)')
plt.tight_layout()
plt.savefig('../poster/Probit_synth/error2newt_iter.png')
plt.show()



print(elbo_newt)
print(w)
print(w_mean)
print(w_post_mean)
print(w_mean_p)
print(np.linalg.norm(z-z_loc))
print(np.linalg.norm(z_post_loc-z_loc))
print(np.linalg.norm(z_loc_p-z_loc))

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
