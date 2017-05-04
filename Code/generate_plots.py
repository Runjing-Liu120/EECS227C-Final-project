import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 14})


# load output of Newton
output0 = open('outputs/Newton_times_probit.pickle', 'rb')
times_newt = np.array(pkl.load(output0))

output1 = open('outputs/Newton_elbo_probit.pickle', 'rb')
elbo_newt  = pkl.load(output1)

output2 = open('outputs/Newton_Wmean_probit.pickle', 'rb')
w_post_mean = pkl.load(output2)
D = len(w_post_mean)

output3 = open('outputs/Newton_pars_probit.pickle', 'rb')
pars_NEWT = pkl.load(output3)


# load output of CAVI
output4 = open('outputs/CAVI_elbo_probit.pickle', 'rb')
elbo_cavi = pkl.load(output4)

output5 = open('outputs/CAVI_Wmean_probit.pickle', 'rb')
w_mean = pkl.load(output5)

output6 = open('outputs/CAVI_times_probit.pickle', 'rb')
times_CAVI = pkl.load(output6)

output7 = open('outputs/CAVI_pars_probit_int.pickle', 'rb')
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
#m = min([elbo_cavi[0],elbo_pxvb[0], elbo_newt[0]])
t0 = 0
plt.plot(times_CAVI[t0:10] - times_CAVI[0], elbo_cavi[t0:10],'ro-')
plt.plot(times_PXVB[t0:20] - times_PXVB[0], elbo_pxvb[t0:20],'bd-')
plt.plot(times_newt[t0:3] - times_newt[0], elbo_newt[t0:3],'g*-')
plt.xlabel("Wall Time (s)") 
plt.ylabel("ELBO")
plt.legend(['CAVI','PXVB','NCG'], loc= 4)
plt.title('ELBO vs Time')
plt.tight_layout()
plt.savefig('../poster/Probit_real/elbo_time.png')
plt.show()




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
"""
# pairwise diffs plot
delta_cavi = [np.linalg.norm(pars_CAVI[i+1][:D] - pars_CAVI[i][:D]) for i in range(len(pars_CAVI)-1)]
delta_pxvb = [np.linalg.norm(pars_PXVB[i+1][:D] - pars_PXVB[i][:D]) for i in range(len(pars_PXVB)-1)]
delta_newt = [np.linalg.norm(pars_NEWT[i+1][:D] - pars_NEWT[i][:D]) for i in range(len(pars_NEWT)-1)]
plt.figure(3)
plt.semilogy(times[1:], delta_cavi,'r')
plt.semilogy(times_PXVB[1:], delta_pxvb,'b')
plt.semilogy(times_newt[1:], delta_newt,'g')
plt.xlabel("Wall Time (s)")
plt.ylabel("$||w^{t+1} - w^t||_2$")
plt.legend(['CAVI','PXVB','NCG'], loc= 4)
plt.title('Difference Between Consecutive Estimates')
plt.tight_layout()
#plt.savefig('../poster/Probit_real/CAVI_PX_convergence.png') 
plt.show()

"""