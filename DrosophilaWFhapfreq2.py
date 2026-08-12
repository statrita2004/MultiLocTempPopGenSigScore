import torch, sys
from scipy import stats
import numpy as np
import pylab as plt
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

### Input which repitition ###
#################################
rep_ind = int(sys.argv[1])
#################################
###### Choose different experimental setups for n_param=2 ########################
h = [[.5, .5]]
# r = [[(0.03/1000000)*(1363924-1314839), (0.03/1000000)*(1372793-1363924)]]
# print(r)

r = [[(0.03/1000000)*(2387650-1541936)]]
n_param = 2
n_rep = 10

ii = 0
###### Load Relative Allele Frequency (Drosophila) Dataset ############
# ## Old Data ##
# from numpy import genfromtxt
# my_data = genfromtxt('Data/Drosophila/freqs.csv', delimiter=',')
# my_data[:,0] = my_data[:,0]/60
# Drosophila_data = []
# for ind in range(10):
#     Drosophila_data.append(my_data[1+7*ind:8+7*ind, :4])
# Drosophila_data = torch.tensor(np.swapaxes(np.array(Drosophila_data), axis1=0, axis2=1))
## New Data ##
Drosophila_data = torch.tensor(np.load('Data/Drosophila/Drosophila.npz')['data']).double()[:,:,[0, 1, 3]]
Drosophila_data_one = Drosophila_data[:,rep_ind,:].reshape(7, 1, 3)

# Unknown initial haplotype frequency
from mysrc.ModelWF import WFUniformDirichlet2 as WFUD
WFUD= WFUD(neg_approx_llhd, population_size=300, generation=60, generation_interval=[10],
                 recomb_param=r[0], data_obs=Drosophila_data_one, n_sample=100, n_param=2)
lpost = lambda x: WFUD.llhd_grad(x) + WFUD.logprior_grad(x)

estimate_lls = np.zeros(shape=(10, 2))
for ind_datarep in range(10):
    data_obs_tmp_numpy = Drosophila_data[:,ind_datarep,:].detach().numpy().squeeze()
    for ind_dim in range(data_obs_tmp_numpy.shape[1]-1):
            nonzero_index = np.argwhere(data_obs_tmp_numpy[:,ind_dim+1]>0).max()
            p_t, p_0 = data_obs_tmp_numpy[nonzero_index,ind_dim+1],data_obs_tmp_numpy[0,ind_dim+1]
            if p_t == 0:
                p_t = 1e-10
            if p_t == 1.0:
                p_t = 1.0 - 1e-10
            if p_t == p_0:
                estimate_lls[ind_datarep, ind_dim] = 0
            if p_t == p_0:
                estimate_lls[ind_datarep, ind_dim] = 0
            else:
                estimate_lls[ind_datarep, ind_dim] = (np.log((p_t * (1-p_0))/(p_0*(1-p_t))))*(2/(10*nonzero_index))
print('LLS estimate:', estimate_lls[rep_ind,:])
#
# ## Posterior Sampling from Scoring Rule Posterior using MH###
# # ####################################################
# x0 = WFUD.transformationtoR(torch.tensor(estimate_lls[rep_ind, :].tolist()+[0.25, 0.25, 0.25, 0.25]))
# from mysrc.MH import MH
# xx = MH(lpost, x0 = x0,
#         sigma = 1e-3 * torch.tensor([[1, 0, 0, 0, 0, 0],
#                                      [0, 1, 0, 0, 0, 0],
#                                      [0, 0, 1, 0, 0, 0],
#                                     [0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1]], dtype=torch.float64), nmoves=100000, return_entire_chain=True, adapt=False, adapt_no=5000)
# xxa = []
# for ind in range(xx.shape[0]):
#     xxa.append(WFUD.invtransformationfromR(xx[ind]).detach().numpy())
# xxa = np.array(xxa)
# np.savez('Results/Drosophila/Drosophila_hap_2_'+str(rep_ind), samples=xxa, samplesR = xx)

xx = np.load('Results/Drosophila/Drosophila_hap_2_'+str(rep_ind)+'.npz')['samples']
burnin = 90000
xx = xx[burnin:,:]
postmean = np.average(xx, axis=0)
print('Posterior Mean estimate:',postmean.tolist())
print('[',str(np.quantile(xx[:,0],0.05))+','+str(np.quantile(xx[:,0],0.95))+']')
print('[',str(np.quantile(xx[:,1],0.05))+','+str(np.quantile(xx[:,1],0.95))+']')
# Contour Plot of the Samples
# print(estimate_lls, postmean)
i,j=0, 1
plt.figure()
bw_method = .9
xmin, xmax = -.5, .5
ymin, ymax = -.5, .5
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:,i].T, xx[:,j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.colorbar()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=5, label='samples')
plt.plot(estimate_lls[rep_ind, :].tolist()[i], estimate_lls[rep_ind, :].tolist()[j], 'bx', markersize=30, label='LLS estimate')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.legend(fontsize=20)
plt.xlabel('$s_1$',fontsize=15)
plt.ylabel('$s_2$',fontsize=15)
plt.savefig('Results/Drosophila/Drosophila_hap_2_'+str(rep_ind)+'_'+str(i)+str(j)+'.png')
plt.close()

i, j = 2,3
#print(i, j)
plt.figure()
bw_method = .9
# xmin, xmax = min(xx[:, i]) - 2* (max(xx[:, i])-min(xx[:, i])), max(xx[:, i]) + 2* (max(xx[:, i])-min(xx[:, i]))
# ymin, ymax = min(xx[:, j]) - 2* (max(xx[:, j])-min(xx[:, j])), max(xx[:, j]) + 2* (max(xx[:, j])-min(xx[:, j]))
xmin, xmax = max(0,min(xx[:, i]) - 2* (max(xx[:, i])-min(xx[:, i]))), min(1,max(xx[:, i]) + 2* (max(xx[:, i])-min(xx[:, i])))
ymin, ymax = max(0,min(xx[:, j]) - 2* (max(xx[:, j])-min(xx[:, j]))), min(1,max(xx[:, j]) + 2* (max(xx[:, j])-min(xx[:, j])))
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:, i].T, xx[:, j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.colorbar()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=5, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.legend(fontsize=20)
plt.xlabel('$h_1$', fontsize=15)
plt.ylabel('$h_2$', fontsize=15)
plt.savefig('Results/Drosophila/Drosophila_hap_2_'+str(rep_ind)+'_'+str(i) + str(j) + '.png')
plt.close()

i, j = 4, 5
#print(i, j)
plt.figure()
bw_method = .9
# xmin, xmax = min(xx[:, i]) - 2* (max(xx[:, i])-min(xx[:, i])), max(xx[:, i]) + 2* (max(xx[:, i])-min(xx[:, i]))
# ymin, ymax = min(xx[:, j]) - 2* (max(xx[:, j])-min(xx[:, j])), max(xx[:, j]) + 2* (max(xx[:, j])-min(xx[:, j]))
xmin, xmax = max(0,min(xx[:, i]) - 2* (max(xx[:, i])-min(xx[:, i]))), min(1,max(xx[:, i]) + 2* (max(xx[:, i])-min(xx[:, i])))
ymin, ymax = max(0,min(xx[:, j]) - 2* (max(xx[:, j])-min(xx[:, j]))), min(1,max(xx[:, j]) + 2* (max(xx[:, j])-min(xx[:, j])))
X, Y = np.mgrid[xmin:xmax:20j, ymin:ymax:20j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([xx[:, i].T, xx[:, j].T])
kernel = stats.gaussian_kde(values, bw_method=bw_method)
Z = np.reshape(kernel(positions).T, X.shape)
CS = plt.contour(X, Y, Z, 30, linestyles='solid')
plt.colorbar()
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
# plt.plot(xx[:,i], xx[:,j], 'k.', markersize=5, label='samples')
plt.plot(postmean[i], postmean[j], 'rx', markersize=30, label='posterior mean')
plt.legend(fontsize=20)
plt.xlabel('$h_3$', fontsize=15)
plt.ylabel('$h_4$', fontsize=15)
plt.savefig('Results/Drosophila/Drosophila_hap_2_'+str(rep_ind)+'_'+str(i) + str(j) + '.png')
plt.close()
