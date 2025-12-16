import torch
from scipy import stats
import numpy as np
import pylab as plt
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF', omega=5).estimate_score_batch

from mysrc.ModelWF import ModelWrightFisher as WF

###### Choose different experimental setups for n_param=3 ########################
#req_chr4_690968" "freq_chr4_692033" "freq_chr4_693728
# r = 0 (window recombination rate is 0)
r = [[0,0]]

hap_freq_K = [[0.0319, 0.1644, 0.1852, 0.1487, 0.011, 0.0254, 0.2716, 0.1619]]

yeast_data_K = torch.tensor([[[0.00, 0.07692308, 0.34782609, 0.20689655]],
[[0.50, 0.4579439, 0.1747573, 0.4519231]],
[[1.00, 0.5079365, 0.1333333, 0.5073529]]], dtype=torch.float64)

hap_freq_S = [[0.1411, 0.5509, 0.0168, 0.0566, 0.12, 0.071, 3e-04, 0.0432]]

yeast_data_S = torch.tensor([[[0.00,  0.20689655, 0.08943089, 0.70642202]],
[[0.50, 0.2376238, 0, 0.7297297]],
[[1.00, 0.4661654, 0, 0.6808511]]], dtype=torch.float64)


n_param = 3
n_rep = 1

WF = WF(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=2000,
        generation=210,generation_interval=[105, 210], recomb_param = r[0],
        haplotype_freq = hap_freq_S[0], dominance_param = None)

from mysrc.ModelWF import WFUniform as WFU
WFU= WFU(neg_approx_llhd, population_size=2000, generation=210, generation_interval=[105, 210],
         haplotype_freq = hap_freq_S[0],
                 recomb_param=r[0], data_obs=yeast_data_S, n_sample=100, n_param=n_param, n_ensemble=10)
lpost = lambda x: WFU.llhd_grad(x) + WFU.logprior_grad(x)

## Compute LLS Estimate ###
##############################################
data_obs_tmp_numpy = yeast_data_S[:,:,:].detach().numpy().squeeze()
estimate_lls = np.zeros(shape=(3))
for ind_dim in range(data_obs_tmp_numpy.shape[1]-1):
    nonzero_index = np.argwhere(data_obs_tmp_numpy[:,ind_dim+1]>0).max()
    p_t, p_0 = 1-data_obs_tmp_numpy[nonzero_index,ind_dim+1],1-data_obs_tmp_numpy[0,ind_dim+1]
    if p_t == p_0:
        estimate_lls[ind_dim] = 0
    else:
        estimate_lls[ind_dim] = (np.log((p_t * (1-p_0))/(p_0*(1-p_t))))*(2/(105*nonzero_index))
print('LLS estimate:',estimate_lls)

######Posterior Sampling from Scoring Rule Posterior using MH###
#####################################################
x0 = WFU.transformationtoR(torch.tensor(estimate_lls))
from mysrc.MH import MH
xx = MH(lpost, x0 = x0,
        sigma = 1e-4 * torch.tensor([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]], dtype=torch.float64), nmoves=1000, return_entire_chain=True, adapt=False)
xxa = []
for ind in range(xx.shape[0]):
    xxa.append(WFU.invtransformationfromR(xx[ind]).detach().numpy())
xxa = np.array(xxa)
np.savez('Results/Yeast/Yeast_S',samples=xxa, samplesR = xx)

xx = np.load('Results/Yeast/Yeast_S.npz')['samples']
burnin = 300
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print('Posterior Mean:', postmean)

# Contour Plot of the Samples
plt.figure()
i,j=0,1
bw_method = .9
xmin, xmax = -.03, .03
ymin, ymax = -.03, .03
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.colorbar()
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=5, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
#plt.plot(s[0][i], s[0][j], 'bx', markersize=30, label='true value')
plt.legend(fontsize=20)
plt.xlabel(r'$s_1$',fontsize=15)
plt.ylabel(r'$s_2$',fontsize=15)
plt.savefig('Results/Yeast/Yeast_S_'+str(i)+str(j)+'.jpg')
plt.close()

# Contour Plot of the Samples
plt.figure()
i,j=1,2
bw_method = .9
xmin, xmax = -.03, .03
ymin, ymax = -.03, .03
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.colorbar()
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.legend(fontsize=20)
plt.xlabel(r'$s_2$',fontsize=15)
plt.ylabel(r'$s_3$',fontsize=15)
plt.savefig('Results/Yeast/Yeast_S_'+str(i)+str(j)+'.jpg')
plt.close()