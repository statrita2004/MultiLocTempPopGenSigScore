import torch
import numpy as np
from scipy import stats
import pylab as plt
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=0.01, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

from mysrc.ModelFDS import ModelWrightFisherFDS as FDS

# ###### Choose different experimental setups for n_param=3 ########################
s = [[0.01, 0.05, 0.07], [0.02, 0.07, 0.05]]
r = [[0.001,0.01],[0.1,0.5]]


n_param = 3
n_rep = 1

FDS = FDS(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=5000,
        generation=100,generation_interval=[10], recomb_param = r[1],
        haplotype_freq = [0.2, 0.2, 0.05, 0.15, 0.05, 0.05, 0.2, 0.1], dominance_param = None)

# Simule datasets for fixed selection coefficient s[ind_s]
data_obs = FDS.forward_simulation(torch.tensor(s[1]), n_data=n_rep)

from mysrc.ModelFDS import FDSUniform as FDSU
FDSU= FDSU(neg_approx_llhd, population_size=5000, generation=100, generation_interval=[10],
                 recomb_param=r[1], haplotype_freq=[0.2, 0.2, 0.05, 0.15, 0.05, 0.05, 0.2, 0.1],
         data_obs=data_obs, n_sample=100, n_param=n_param, n_ensemble= 10)
lpost = lambda x: FDSU.llhd_grad(x) + FDSU.logprior_grad(x)

## Posterior Sampling from Scoring Rule Posterior using MH###
####################################################
x0 = FDSU.transformationtoR(torch.tensor([0.0, 0.0, 0.0]))
from mysrc.MH import MH
xx = MH(lpost, x0 = x0,
        sigma = 1e-2 * torch.tensor([[1, 0, 0], [0, 1, 0],
        [0, 0, 1]], dtype=torch.float64), nmoves=5000, return_entire_chain=True, adapt=False)

for ind in range(xx.shape[0]):
    xx[ind] = FDSU.invtransformationfromR(xx[ind])
np.savez('Results/FDS/MH_FDS_3_1', samples=xx)


xx = np.load('Results/FDS/MH_FDS_3_1.npz')['samples']
burnin = 4000
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print(postmean)
#
# Contour Plot of the Samples
plt.figure()
i,j=0,1
bw_method = .9
xmin, xmax = -.2, .2
ymin, ymax = -.2, .2
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot(xx[:,i], xx[:,j], 'kx-')
# plt.colorbar()
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.plot(s[1][i], s[1][j], 'bx', markersize=30, label='true value')
# plt.legend(fontsize=20)
plt.xlabel(r'$s_1$',fontsize=15)
plt.ylabel(r'$s_2$',fontsize=15)
plt.savefig('Results/FDS/MH_FDS_3_1_'+str(i)+str(j)+'.png')
plt.close()

i,j=1,2
plt.figure()
bw_method = .9
xmin, xmax = -.2, .2
ymin, ymax = -.2, .2
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot(xx[:,i], xx[:,j], 'kx-')
#plt.colorbar()
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.plot(s[1][i], s[1][j], 'bx', markersize=30, label='true value')
plt.xlabel(r'$s_2$',fontsize=15)
plt.ylabel(r'$s_3$',fontsize=15)
plt.savefig('Results/FDS/MH_FDS_3_1_'+str(i)+str(j)+'.png')
plt.close()

