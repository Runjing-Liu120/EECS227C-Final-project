# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:17:12 2017

@author: Haiying Liang

Probit regression CAVI
"""
import numpy as np
from scipy.stats import norm
import warnings

# compute mean and variance of truncated normal
# z should have variance 1
def trunc_Normal(z_loc, t): 
    
    Z_ratio1 = norm.pdf(-z_loc)/(1 - norm.cdf(-z_loc))
    Z_ratio2 = norm.pdf(-z_loc)/norm.cdf(-z_loc)
    
    if any(np.isinf(Z_ratio1)) or any(np.isinf(Z_ratio2)):
        print('dividing by 0 in computing truncated normal parameters')

    #Z_ratio1[np.isinf(Z_ratio1)] = 0
    #Z_ratio2[np.isinf(Z_ratio1)] = 0
            
    # compute mean of z
    z_trunc_mean = (t>0) * ( z_loc + Z_ratio1 ) \
                + (t<0) * ( z_loc - Z_ratio2 )
    
    z_trunc_var = (t>0) * (1 - z_loc * Z_ratio1 - Z_ratio1**2)\
                + (t<0) * (1 + z_loc * Z_ratio2 - Z_ratio2**2)\
    
    return(z_trunc_mean, z_trunc_var)

## CAVI updates
def probit_CAVI(X, t, v_0, w_mean, w_var, z_loc, z_var):
    N = np.shape(X)[1]
    D = np.shape(X)[0]

    # update location parameter of z       
    z_loc = np.dot(w_mean, X)
    
    # compute mean of z
    [z_trunc_mean, z_trunc_var] = trunc_Normal(z_loc, t)
    
    w_var = np.linalg.inv(np.dot(X,X.T) + (1/v_0) * np.identity(D))
    w_mean = np.dot(w_var, np.dot(X, z_trunc_mean))
    
    return(w_mean, w_var, z_loc, z_trunc_mean)
    

# reparametrization step
def probit_reparam(X, t, v_0, w_mean, w_var, z_loc, z_var):
    N = np.shape(X)[1]
    D = np.shape(X)[0]
    
    # expectation of ww^T
    exp_w2 = w_var + np.outer(w_mean, w_mean)
     
    c2 = 1/v_0 * (np.dot(w_mean, w_mean) + np.trace(w_var)) # eq 13 in PX-VB paper
    [z_trunc_mean, z_trunc_var] = trunc_Normal(z_loc, t)
    
    for n in range(N):
        c2 += (z_trunc_mean[n]**2 + z_trunc_var[n]) - \
                2*z_trunc_mean[n]*np.dot(w_mean, X[:,n]) \
                + np.dot(X[:,n],np.dot(exp_w2,X[:,n]))
                
    c2 = c2 * 1/(N+D)
    
    w_mean = w_mean/np.sqrt(c2)
    w_var = w_var/c2
    
    return(w_mean, w_var)

def probit_Gibbs(X, t, v_0, burn, iterations):
    N = np.shape(X)[1]
    D = np.shape(X)[0]
    
    # intializations
    w_Gibbs = np.random.multivariate_normal(np.zeros(D), np.identity(D) )
    z_Gibbs = np.random.normal(0, 1, N)
    
    z_post_mean = np.zeros(N)
    w_post_mean = np.zeros(D)
    n_samples = 0
    
    
    for i in range(iterations): 
                    
        # draw w
        w_var = np.linalg.inv(np.dot(X,X.T) + 1/v_0 * np.identity(D))
        w_mean = np.dot(w_var , np.dot(X, z_Gibbs))
        w_Gibbs = np.random.multivariate_normal(w_mean, w_var)
                
        # draw z_n
        for n in range(N):
            z_Gibbs[n] = np.random.normal(np.dot(X[:,n], w_Gibbs), 1)
            
            counter = 0
            while np.sign(z_Gibbs[n]) != t[n]:
                z_Gibbs[n] = np.random.normal(np.dot(X[:,n], w_Gibbs),1)
                counter = counter+1 
                
                #if (counter >= 100):
                #    print('Gibbs sampler is stuck, n=', n)
                #    print(np.dot(X[:,n], w_Gibbs), t[n])
                #    print(z_Gibbs[n])

        if i >= burn: 
            
            z_post_mean = n_samples/(n_samples+1) * z_post_mean \
                        + 1/(n_samples+1) * z_Gibbs
            w_post_mean = n_samples/(n_samples+1) * w_post_mean \
                        + 1/(n_samples+1) * w_Gibbs

            n_samples += 1
            print(i)
        
    return(w_post_mean, z_post_mean)