import torch
import numpy as np
from scipy import stats
import pylab as plt
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=0.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

from mysrc.ModelWF import ModelWrightFisher as WF

# ###### Choose different experimental setups for n_param=2 ########################
s = [[0.02, 0.02], [0.02, 0.07], [0.02,0.09], [0.02,0.05]]
r = [[0],[1e-6],[1e-2],[0.1],[0.5]]
n_param = 2
n_rep = 1
WF = WF(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=5000,
        generation=100,generation_interval=[10], recomb_param = r[2],
        haplotype_freq = [0.4, 0.1, 0.1, 0.4], dominance_param = None)

# Simule datasets for fixed selection coefficient s[ind_s]
data_obs = WF.forward_simulation(torch.tensor(s[1]), n_data=n_rep)


from mysrc.ModelWF import WFUniform as WFU
WFU= WFU(neg_approx_llhd, population_size=5000, generation=100, generation_interval=[10],
                 recomb_param=r[2], haplotype_freq=[0.4, 0.1, 0.1, 0.4],
         data_obs=data_obs, n_sample=100, n_param=2, n_ensemble= 10)
lpost = lambda x: WFU.llhd_grad(x) + WFU.logprior_grad(x)

print(lpost(WFU.transformationtoR(torch.tensor([0, 0]))))

# Posterior Sampling from Scoring Rule Posterior using MH###
####################################################
x0 = WFU.transformationtoR(torch.tensor([0.0, 0.0]))
from mysrc.MH import MH
xx = MH(lpost, x0 = x0,
        sigma = 1e-4 * torch.tensor([[1, 0],
        [0,  1]], dtype=torch.float64), nmoves=1000, return_entire_chain=True, adapt=False)

for ind in range(xx.shape[0]):
    xx[ind] = WFU.invtransformationfromR(xx[ind])

np.savez('Results/WF/MH_2', samples=xx)


xx = np.load('Results/WF/MH_2.npz')['samples']
burnin = 200
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print(postmean)

# Contour Plot of the Samples
i,j=0,1
bw_method = .9
xmin, xmax = -.1, .1
ymin, ymax = -.1, .1
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.colorbar()
plt.plot(postmean[0], postmean[1], 'rx', markersize=30, label='posterior mean')
plt.plot(s[1][0], s[1][1], 'bx', markersize=30, label='true value')
plt.legend(fontsize=20)
plt.xlabel(r'$s_1$',fontsize=15)
plt.ylabel(r'$s_2$',fontsize=15)
plt.savefig('Results/WF/MH_2_'+str(i)+str(j)+'.jpg')