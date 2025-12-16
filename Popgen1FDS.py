import torch
import numpy as np
from scipy import stats
import pylab as plt
from mysrc.scoring_rules import SignatureKernel  # Your ES module
neg_approx_llhd = SignatureKernel(rbf_sigma=0.01, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

from mysrc.ModelFDS import ModelWrightFisherFDS as FDS

# ###### Choose different experimental setups for n_param=2 ########################
s = [[0.02], [0.07], [0.09], [0.05]]
r = [[0],[1e-6],[1e-2],[0.1],[0.5]]
n_param = 1
n_rep = 1
FDS = FDS(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=5000,
        generation=100,generation_interval=[10], recomb_param = r[0],
        haplotype_freq = [0.4, 0.6], dominance_param = None)

# Simule datasets for fixed selection coefficient s[ind_s]
data_obs = FDS.forward_simulation(torch.tensor(s[1]), n_data=n_rep)

from mysrc.ModelFDS import FDSUniform as FDSU
FDSU= FDSU(neg_approx_llhd, population_size=5000, generation=100, generation_interval=[10],
                 recomb_param=r[2], haplotype_freq=[0.4, 0.6],
         data_obs=data_obs, n_sample=10, n_param=1, n_ensemble= 10)
lpost = lambda x: FDSU.llhd_grad(x) + FDSU.logprior_grad(x)

## Posterior Sampling from Scoring Rule Posterior using MH###
####################################################
x0 = FDSU.transformationtoR(torch.tensor([0]))
from mysrc.MH import MH
xx = MH(lpost, x0 = x0,
        sigma = torch.tensor(1e-2), nmoves=2000, return_entire_chain=True, adapt=False)

for ind in range(xx.shape[0]):
    xx[ind] = FDSU.invtransformationfromR(xx[ind])

np.savez('Results/FDS/MH_FDS_1', samples=xx)

xx = np.load('Results/FDS/MH_FDS_1.npz')['samples']
print(xx.shape)
burnin = 500
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print(postmean)

plt.figure()
xmin, xmax = 0, 0.1
positions = np.linspace(xmin, xmax, 100)
gaussian_kernel = stats.gaussian_kde(xx[:,0], bw_method=0.9)
plt.plot(positions, gaussian_kernel(positions), color='k', linestyle='solid', lw="1",
                                      alpha=1, label="Posterior Density")
plt.plot([postmean[0], postmean[0]], [0, 15], 'r', label='posterior mean')
plt.plot([s[1][0], s[1][0]], [0, 15], 'b', label='True value')
plt.legend()
plt.savefig('Results/FDS/MH_FDS_1.jpg')
plt.close()
