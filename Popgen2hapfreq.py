import torch
from scipy import stats
import numpy as np
import pylab as plt
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

from mysrc.ModelWF import ModelWrightFisherHapFreq as WF

# ###### Choose different experimental setups for n_param=2 ########################
s = [[0.02, 0.02], [0.02, 0.07], [0.02,0.09], [0.02,0.05]]
r = [[0],[1e-6],[1e-2],[0.1],[0.5]]
hap_freq = [[0.4, 0.1, 0.1, 0.4]]

n_param = 2
n_rep = 1
WF = WF(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=5000,
        generation=100, generation_interval=[10], recomb_param = r[1])

# Simule datasets for fixed selection coefficient s[ind_s]
data_obs = WF.forward_simulation(torch.tensor(s[1]+hap_freq[0]), n_data=n_rep)

from mysrc.ModelWF import WFUniformDirichlet2 as WFUD
WFUD= WFUD(neg_approx_llhd, population_size=5000, generation=100, generation_interval=[10],
                 recomb_param=r[1], data_obs=data_obs, n_sample= 100, n_param=2)
lpost = lambda x: WFUD.llhd_grad(x) + WFUD.logprior_grad(x)

## Posterior Sampling from Scoring Rule Posterior using MH###
#####################################################
x0 = WFUD.transformationtoR(torch.tensor([0.0, 0.0, 0.25, 0.25, 0.25, 0.25]))
from mysrc.MH import MH
xx = MH(lpost, x0 = x0,
        sigma = 1e-3 * torch.tensor([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]], dtype=torch.float64), nmoves=100, return_entire_chain=True, adapt=False, adapt_no=5000)
xxa = []
for ind in range(xx.shape[0]):
    xxa.append(WFUD.invtransformationfromR(xx[ind]).detach().numpy())
xxa = np.array(xxa)
postmean = np.average(xxa, axis=0)
print(postmean)
np.savez('Results/WF/MH_2_hap', samples=xxa, samplesR = xx)

xx = np.load('Results/WF/MH_2_hap.npz')['samples']
burnin = 10000
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print(postmean)

# Contour Plot of the Samples
plt.figure()
i,j=0, 1
bw_method = .9
xmin, xmax = -0.1, 0.1
ymin, ymax = -0.1, 0.1
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.colorbar()
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=1, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.plot(s[1][0], s[1][1], 'bx', markersize=30, label='true value')
#plt.plot(hap_freq[0][2], hap_freq[0][3], 'bx', markersize=30, label='true value')
#plt.legend(fontsize=20)
plt.xlabel(r'$s_1$',fontsize=15)
plt.ylabel(r'$s_2$',fontsize=15)
plt.savefig('Results/WF/MH_2_hap_'+str(i)+str(j)+'.jpg')
plt.close()

# Contour Plot of the Samples
plt.figure()
i,j=2,3
bw_method = .9
xmin, xmax = 0.0, 0.6
ymin, ymax = 0.0, 0.6
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.colorbar()
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=1, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
#plt.plot(s[1][0], s[1][1], 'bx', markersize=30, label='true value')
plt.plot(hap_freq[0][0], hap_freq[0][1], 'bx', markersize=30, label='true value')
#plt.legend(fontsize=20)
plt.xlabel(r'$h_1$',fontsize=15)
plt.ylabel(r'$h_2$',fontsize=15)
plt.savefig('Results/WF/MH_2_hap_'+str(i)+str(j)+'.jpg')
plt.close()

# Contour Plot of the Samples
plt.figure()
i,j=4,5
bw_method = .9
xmin, xmax = 0.0, 0.6
ymin, ymax = 0.0, 0.6
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.colorbar()
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=1, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
#plt.plot(s[1][0], s[1][1], 'bx', markersize=30, label='true value')
plt.plot(hap_freq[0][2], hap_freq[0][3], 'bx', markersize=30, label='true value')
#plt.legend(fontsize=20)
plt.xlabel(r'$h_3$',fontsize=15)
plt.ylabel(r'$h_4$',fontsize=15)
plt.savefig('Results/WF/MH_2_hap_'+str(i)+str(j)+'.jpg')
plt.close()