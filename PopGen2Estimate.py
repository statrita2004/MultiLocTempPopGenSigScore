import torch
import numpy as np
import time
from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

###### Choose different experimental setups for n_param=2 ########################
s = [[0.02, 0.02], [0.02, 0.07], [0.02,0.09], [0.02,0.05]]
hap_freq = [[0.4, 0.1, 0.1, 0.4]]
h = [[.5, .5, .5]]
epistasis_alpha = [[8]]
r = [[0],[1e-6],[1e-2],[0.1],[0.5]]

n_param = 2
n_rep = 10
burnin = 200

estimate_mcmc = np.zeros(shape=(len(s), len(r), n_rep, n_param))
MSE_mean = np.zeros(shape=(len(s), len(r)))
MSE_std = np.zeros(shape=(len(s), len(r)))
runtime = np.zeros(shape=(len(s), len(r), n_rep))

estimate_lls = np.zeros(shape=(len(s), len(r), n_rep, n_param))
MSE_mean_lls = np.zeros(shape=(len(s), len(r)))
MSE_std_lls = np.zeros(shape=(len(s), len(r)))

for ind_s in range(len(s)):
        for ind_r in range(len(r)):
                print(ind_s, ind_r)
                print('Running inference: true_s,', s[ind_s], 'true_r', r[ind_r])
                from mysrc.ModelWF import ModelWrightFisher as WF
                WF = WF(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=5000,
                        generation=100,generation_interval=[10], recomb_param = r[ind_r],
                        haplotype_freq = [0.4, 0.1, 0.1, 0.4], dominance_param = None)

                # Simule n_data=10 datasets for fixed selection coefficient s[ind_s]
                data_obs = WF.forward_simulation(torch.tensor(s[ind_s]), n_data=n_rep)

                for ind_rep in range(n_rep):
                        data_obs_tmp_numpy = data_obs[:,ind_rep:ind_rep+1,:].detach().numpy().squeeze()
                        for ind_dim in range(data_obs_tmp_numpy.shape[1]-1):
                                nonzero_index = np.argwhere(data_obs_tmp_numpy[:,ind_dim+1]>0).max()
                                p_t, p_0 = 1-data_obs_tmp_numpy[nonzero_index,ind_dim+1],1-data_obs_tmp_numpy[0,ind_dim+1]
                                estimate_lls[ind_s, ind_r, ind_rep,ind_dim] = (np.log((p_t * (1-p_0))/(p_0*(1-p_t))))*(2/(50*nonzero_index))
                SE_lls = np.sqrt((1/WF.L) * np.sum(np.square(estimate_lls[ind_s, ind_r,:,:] - s[ind_s]), axis=1))
                MSE_mean_lls[ind_s, ind_r], MSE_std_lls[ind_s, ind_r] = np.mean(SE_lls), np.std(SE_lls)
                print(MSE_mean_lls, MSE_std_lls)
                np.savez('Results/WF/selcof_2_lls', estimate_lls=estimate_lls, MSE_mean_lls=MSE_mean_lls, MSE_std_lls=MSE_std_lls)

                for ind_rep in range(n_rep):
                        data_obs_tmp = data_obs[:,ind_rep:ind_rep+1,:]

                        from mysrc.ModelWF import WFUniform as WFU

                        WFU = WFU(neg_approx_llhd, population_size=5000, generation=100, generation_interval=[10],
                                  recomb_param=r[ind_r], haplotype_freq=[0.4, 0.1, 0.1, 0.4],
                                  data_obs=data_obs, n_sample=100, n_param=n_param)
                        lpost = lambda x: WFU.llhd_grad(x) + WFU.logprior_grad(x)

                        x0 = WFU.transformationtoR(torch.tensor([0.0, 0.0]))
                        from mysrc.MH import MH
                        start_t = time.time()
                        xx = MH(lpost, x0 = x0,
                                sigma = 1e-4 * torch.tensor([[1, 0], [0, 1]], dtype=torch.float64), nmoves=1000, return_entire_chain=True, adapt=True)
                        for ind in range(xx.shape[0]):
                            xx[ind] = WFU.invtransformationfromR(xx[ind])
                        xx = xx[burnin:, :]
                        postmean = np.average(xx, axis=0)
                        end_t = time.time()
                        estimate_mcmc[ind_s, ind_r, ind_rep, :] = postmean
                        runtime[ind_s, ind_r, ind_rep] = end_t - start_t
                        print(postmean, end_t-start_t)
                print('estimate_mcmc', estimate_mcmc[ind_s, ind_r,:,:])
                SE_mcmc = np.sqrt((1/WF.L) * np.sum(np.square(estimate_mcmc[ind_s, ind_r,:,:] - s[ind_s]), axis=1))
                print('SE', SE_mcmc)
                MSE_mean[ind_s, ind_r], MSE_std[ind_s, ind_r] = np.mean(SE_mcmc), np.std(SE_mcmc)
                print(MSE_mean, MSE_std)
                np.savez('Results/WF/selcof_2_mcmc', estimate_da=estimate_mcmc, MSE_mean=MSE_mean, MSE_std=MSE_std, runtime=runtime)