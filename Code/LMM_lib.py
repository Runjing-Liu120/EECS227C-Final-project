# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:35:36 2017

@author: Haiying Liang
"""
import numpy as np
from scipy.stats import norm
from copy import deepcopy
import matplotlib.pyplot as plt

def lmm_CAVI(X, y, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K):

    # update mu parameters
    for g in range(NG): 
        mu_mean[g] = mu_var * (1/prior_var['y']) * np.sum(y[g*N:(g+1)*N] - np.dot(X[:,g*N:(g+1)*N].T,beta_mean)) 
        
                
                
    # update beta
    # beta_var = np.linalg.inv(1/prior['beta']*np.identity(K) \
        #                          + 1/prior_var['y'] * np.dot(X,X.T))
    NObs = N*NG
    y_g_vec = np.array([ g for g in range(NG) for n in range(N) ])
    canonical_beta_mean = (np.dot(X, y) - sum(mu_mean[n//N]*X[:,n] for n in range(NObs))) / prior_var['y']
    beta_mean = np.dot(beta_var, canonical_beta_mean)
    
    return(beta_mean, mu_mean)


from autograd import grad, hessian, jacobian, hessian_vector_product
import autograd.numpy as anp
import autograd.scipy as asp
from scipy import optimize
from copy import deepcopy as copy
from time import time

def lmm_Newton(X, y, beta_mean, beta_var, mu_mean, mu_var, prior_var, NG,N,K,maxiter=5000):
    par_init = np.concatenate((beta_mean, mu_mean))
    kl_wrapper = KLWrapper(par_init, X, y, beta_var, mu_var, prior_var, NG,N,K)
    
    elbo = []
    times= []
    t0 = time()
    def callbackF(par):
        elbo.append(-kl_wrapper.kl(par, X, y, beta_var, mu_var, prior_var, NG,N,K))
        times.append(time()-t0)

    vb_opt = optimize.minimize(
        lambda par: kl_wrapper.kl(par, X, y, beta_var, mu_var, prior_var, NG,N,K),
        par_init, method='trust-ncg', jac=kl_wrapper.kl_grad, hessp=kl_wrapper.kl_hvp,
        callback=callbackF,
        tol=1e-6, options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-9 })
    return vb_opt, times, elbo

def get_elbo(par, X, y, beta_var, mu_var, prior_var, NG,N,K):
    NObs = N*NG
    beta_mean = par[:K]
    mu_mean   = par[K:]
    term1 = - (anp.trace(beta_var) + anp.dot(beta_mean,beta_mean)) / (2.*prior_var['beta'])
    term2 = - sum(mu_var+mu_mean[g]**2 for g in range(NG)) / (2.*prior_var['mu']) \
            - sum(mu_var+mu_mean[n // N]**2 for n in range(NObs)) / (2.*prior_var['y'])
    term3 = sum(mu_mean[n // N]*(y[n]-anp.dot(X[:,n],beta_mean)) for n in range(NObs)) / prior_var['y']
    #term4 = (anp.dot(y,anp.dot(X.T,beta_mean))-anp.linalg.norm(anp.dot(X.T,beta_mean))**2) / (2.*prior_var['y'])

    #print "A: ", (anp.dot(y,anp.dot(X.T,beta_mean))-anp.linalg.norm(anp.dot(X.T,beta_mean))**2) / (2.*prior_var['y'])
    term4 = (2.*anp.dot(y,anp.dot(X.T,beta_mean)) - anp.trace(anp.dot(X.T,anp.dot(beta_var+anp.outer(beta_mean,beta_mean),X)))) / (2.*prior_var['y'])
    #print anp.trace(anp.dot(X.T,anp.dot(beta_var, X)))
    #print "B: ", (anp.dot(y,anp.dot(X.T,beta_mean)) - anp.trace(anp.dot(X.T,anp.dot(beta_var+anp.outer(beta_mean,beta_mean),X)))) / (2.*prior_var['y'])
    term5 = NG*anp.log(2.*anp.pi*anp.e*mu_var) / 2. + anp.log((2.*anp.pi*anp.e)**K*anp.linalg.det(beta_var)) / 2.
    return term1 + term2 + term3 + term4 + term5

class KLWrapper(object):
    def __init__(self, par, X, y, beta_var, mu_var, prior_var, NG,N,K):
        self.kl_grad = grad(lambda p : self.kl(p, X, y, beta_var, mu_var, prior_var, NG,N,K))
        self.kl_hess = hessian(lambda p : self.kl(p, X, y, beta_var, mu_var, prior_var, NG,N,K))
        self.kl_hvp  = hessian_vector_product(lambda p : self.kl(p, X, y, beta_var, mu_var, prior_var, NG,N,K))

    def kl(self, par, X, y, beta_var, mu_var, prior_var, NG,N,K,verbose=False):
        # kl up to a constant
        kl = -get_elbo(par, X, y, beta_var, mu_var, prior_var, NG,N,K)
        if verbose: print kl
        return kl
