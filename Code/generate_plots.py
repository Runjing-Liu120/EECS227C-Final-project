import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


# load output of Newton
output0 = open('outputs/Newton_times_probit.pickle', 'rb')
times_newt = pkl.load(output0)

output1 = open('outputs/Newton_elbo_probit.pickle', 'rb')
elbo_newt  = pkl.load(output1)

output2 = open('outputs/Newton_Wmean_probit.pickle', 'rb')
w_post_mean = pkl.load(output2)
D = len(w_post_mean)

#output3 = open('outputs/Newton_pars_probit.pickle', 'rb')
#pars_newt = pkl.load(output3)


# load output of CAVI
output4 = open('outputs/CAVI_elbo_probit.pickle', 'rb')
elbo_cavi = pkl.load(output4)

output5 = open('outputs/CAVI_Wmean_probit.pickle', 'rb')
w_mean = pkl.load(output5)

output6 = open('outputs/CAVI_times_probit.pickle', 'rb')
times_CAVI = pkl.load(output6)

output7 = open('outputs/CAVI_pars_probit.pickle', 'rb')
pars_CAVI = pkl.load(output7)


# load output of PXVB
output8 = open('outputs/PXVB_elbo_probit.pickle', 'rb')
elbo_pxvb = pkl.load(output8)

output9 = open('outputs/PXVB_Wmean_probit.pickle', 'rb')
w_mean = pkl.load(output9)

output10 = open('outputs/PXVB_times_probit.pickle', 'rb')
times_PXVB = pkl.load(output10)

output11 = open('outputs/PXVB_pars_probit.pickle', 'rb')
pars_PXVB = pkl.load(output11)



# elbo plot
plt.figure(1)
m = min([elbo_cavi[0],elbo_pxvb[0], elbo_newt[0]])
plt.plot(times_CAVI[:10], elbo_cavi[:10],'ro-')
plt.plot(times_PXVB[:20], elbo_pxvb[:20],'bd-')
plt.plot(times_newt[:3], elbo_newt[:3],'g*-')
plt.xlabel("Time (seconds)") 
plt.ylabel("ELBO")
plt.legend(['CAVI','PXVB','NCG'], loc= 4)
plt.title('Objective Value over Time')
plt.savefig('../poster/Probit_real/elbo_time.png')

"""
# dist to minimizer plot
diff_cavi = [np.linalg.norm(par[:D] - w_post_mean) for par in pars_CAVI]
diff_pxvb = [np.linalg.norm(par[:D] - w_post_mean) for par in pars_PXVB]
plt.figure(2)
plt.plot(diff_cavi,'r')
plt.plot(diff_pxvb,'b')
plt.xlabel("iter") 
plt.ylabel("$\|w^t - w^*\|$")
plt.legend(['CAVI','PXVB'], loc= 4)
plt.title('Distance to Newton solution')
"""

# pairwise diffs plot
delta_cavi = [np.linalg.norm(pars_CAVI[i+1][:D] - pars_CAVI[i][:D]) for i in range(len(pars_CAVI)-1)]
delta_pxvb = [np.linalg.norm(pars_PXVB[i+1][:D] - pars_PXVB[i][:D]) for i in range(len(pars_PXVB)-1)]
plt.figure(3)
plt.semilogy(times_CAVI[1:], delta_cavi,'r')
plt.semilogy(times_PXVB[1:], delta_pxvb,'b')
plt.xlabel("Time (seconds)")
plt.ylabel("$||w^{t+1} - w^t||_2$")
plt.legend(['CAVI','PXVB'], loc= 4)
plt.title('Difference Between Consecutive Estimates')
plt.savefig('../poster/Probit_real/CAVI_PX_convergence.png') 
plt.show()
