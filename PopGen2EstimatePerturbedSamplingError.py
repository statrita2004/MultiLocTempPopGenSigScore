import torch
from scipy.stats import binom, poisson
import numpy as np
import time
from backends import BackendMPI, BackendDummy
#backend = BackendMPI()
backend = BackendDummy()

from mysrc.scoring_rules import SignatureKernel
neg_approx_llhd = SignatureKernel(rbf_sigma=.1, dyadic_order = 1, keep_time = True, cumsum=False, static_kernel_name='RBF').estimate_score_batch

###### Choose different experimental setups for n_param=2 ########################
s = [[0.02, 0.07]]
hap_freq = [[0.4, 0.1, 0.1, 0.4]]
h = [[.5, .5, .5]]
epistasis_alpha = [[8]]
r = [[1e-6]]
lamb = [10000]#, 100, 200, 300, 400]

n_param = 2
n_rep = 1
burnin = 1000
n_gen_int = 10
n_gen = 100
n_pop_size = 5000

Data = torch.zeros(size=(len(lamb), len(s), len(r), n_gen_int+1, n_rep, n_param+1), dtype=torch.float64)
Data_perturb = torch.zeros(size=(len(lamb), len(s), len(r), n_gen_int+1, n_rep, n_param+1), dtype=torch.float64)

estimate_mcmc = np.zeros(shape=(len(lamb), len(s), len(r), n_rep, n_param))
MSE_mean_mcmc = np.zeros(shape=(len(lamb),len(s), len(r)))
MSE_std_mcmc = np.zeros(shape=(len(lamb),len(s), len(r)))
runtime_mcmc = np.zeros(shape=(len(lamb),len(s), len(r), n_rep))

estimate_lls = np.zeros(shape=(len(lamb),len(s), len(r), n_rep, n_param))
MSE_mean_lls = np.zeros(shape=(len(lamb),len(s), len(r)))
MSE_std_lls = np.zeros(shape=(len(lamb),len(s), len(r)))
runtime_lls = np.zeros(shape=(len(lamb),len(s), len(r), n_rep))

for ind_lamb in range(len(lamb)):
        for ind_s in range(len(s)):
                for ind_r in range(len(r)):
                        print(ind_lamb, ind_s, ind_r)
                        print('Running inference: true_lamb,', lamb[ind_lamb], 'true_s,', s[ind_s], 'true_r,', r[ind_r])
                        from mysrc.ModelWF import ModelWrightFisher as WF
                        WF = WF(loss_fn = neg_approx_llhd, n_parameter = n_param, population_size=n_pop_size,
                                generation=n_gen,generation_interval=[n_gen_int], recomb_param = r[ind_r],
                                haplotype_freq = [0.4, 0.1, 0.1, 0.4], dominance_param = None)

                        # Simule n_data=10 datasets for fixed selection coefficient s[ind_s]
                        data_obs = WF.forward_simulation(torch.tensor(s[ind_s]), n_data=n_rep)
                        data_obs_perturb = data_obs.clone()
                        N = poisson.rvs(lamb[ind_lamb], size=data_obs[:,:,1:].shape)
                        A = binom.rvs(N, data_obs[:,:,1:]) / N
                        data_obs_perturb[:, :,1:] = torch.tensor(A, dtype=torch.float64)
                        print(data_obs_perturb, data_obs)
                        Data[ind_lamb, ind_s, ind_r, :, :, :] = data_obs
                        Data_perturb[ind_lamb, ind_s, ind_r, :, :, :] = data_obs_perturb

                        np.savez('Results/WF/Data_perturb_samplingerror', Data = Data, Data_perturb = Data_perturb)

                        for ind_rep in range(n_rep):
                                start_t = time.time()
                                data_obs_tmp_numpy = data_obs_perturb[:,ind_rep:ind_rep+1,:].detach().numpy().squeeze()
                                #print(data_obs_tmp_numpy)
                                for ind_dim in range(data_obs_tmp_numpy.shape[1]-1):
                                        nonzero_index = np.argwhere(data_obs_tmp_numpy[:,ind_dim+1]>0).max()
                                        p_t, p_0 = data_obs_tmp_numpy[nonzero_index,ind_dim+1],data_obs_tmp_numpy[0,ind_dim+1]
                                        if p_t == 0:
                                                p_t=1e-10
                                        if p_t == 1.0:
                                                p_t= 1.0 - 1e-10
                                        if p_t == p_0:
                                                estimate_lls[ind_lamb, ind_s, ind_r, ind_rep,ind_dim] = 0
                                        else:
                                                estimate_lls[ind_lamb, ind_s, ind_r, ind_rep,ind_dim] = (np.log((p_t * (1-p_0))/(p_0*(1-p_t))))*(2/(10*nonzero_index))
                                        # print(p_t, p_0, estimate_lls[ind_lamb, ind_s, ind_r, ind_rep,ind_dim])
                                end_t = time.time()
                                runtime_lls[ind_lamb, ind_s, ind_r, ind_rep] = end_t - start_t
                        SE_lls = np.sqrt((1/WF.L) * np.sum(np.square(estimate_lls[ind_lamb,ind_s, ind_r,:,:] - s[ind_s]), axis=1))
                        MSE_mean_lls[ind_lamb,ind_s, ind_r], MSE_std_lls[ind_lamb,ind_s, ind_r] = np.mean(SE_lls), np.std(SE_lls)
                        print(MSE_mean_lls, MSE_std_lls)
                        np.savez('Results/WF/selcof_2_lls_perturb_samplingerror', estimate_lls=estimate_lls, MSE_mean_lls=MSE_mean_lls, MSE_std_lls=MSE_std_lls, runtime_lls=runtime_lls)

                        def MCMC_temp_function(ind_rep):
                                data_obs_tmp = data_obs_perturb[:, ind_rep:ind_rep + 1, :]

                                from mysrc.ModelWF import WFUniform as WFU

                                WFU = WFU(neg_approx_llhd, population_size=n_pop_size, generation=n_gen, generation_interval=[n_gen_int],
                                          recomb_param=r[ind_r], haplotype_freq=[0.4, 0.1, 0.1, 0.4],
                                          data_obs=data_obs_tmp, n_sample=100, n_param=n_param)
                                lpost = lambda x: WFU.llhd_grad(x) + WFU.logprior_grad(x)

                                x0 = WFU.transformationtoR(torch.tensor([0.0, 0.0]))
                                from mysrc.MH import MH
                                start_t = time.time()
                                xx = MH(lpost, x0=x0,
                                        sigma=1e-4 * torch.tensor([[1, 0], [0, 1]], dtype=torch.float64), nmoves=2000,
                                        return_entire_chain=True, adapt=True)
                                for ind in range(xx.shape[0]):
                                        xx[ind] = WFU.invtransformationfromR(xx[ind])
                                xx = xx[burnin:, :]
                                postmean = np.average(xx, axis=0)
                                end_t = time.time()
                                return postmean, end_t-start_t

                        rep_array = [ind_rep for ind_rep in range(n_rep)]
                        rep_array_pds = backend.parallelize(rep_array)
                        MCMC_temp_pds = backend.map(MCMC_temp_function, rep_array_pds)
                        postmean_rtime_tuple = backend.collect(MCMC_temp_pds)
                        postmean, r_time = zip(*postmean_rtime_tuple)

                        estimate_mcmc[ind_lamb,ind_s, ind_r, :, :] = np.array(postmean)
                        runtime_mcmc[ind_lamb,ind_s, ind_r, :] = np.array(r_time)

                        print('estimate_mcmc', estimate_mcmc[ind_lamb,ind_s, ind_r,:,:])
                        SE_mcmc = np.sqrt((1/WF.L) * np.sum(np.square(estimate_mcmc[ind_lamb,ind_s, ind_r,:,:] - s[ind_s]), axis=1))
                        print('SE', SE_mcmc)
                        MSE_mean_mcmc[ind_lamb,ind_s, ind_r], MSE_std_mcmc[ind_lamb,ind_s, ind_r] = np.mean(SE_mcmc), np.std(SE_mcmc)
                        print(MSE_mean_mcmc, MSE_std_mcmc)
                        np.savez('Results/WF/selcof_2_mcmc_perturb_samplingerror', estimate_da=estimate_mcmc, MSE_mean=MSE_mean_mcmc, MSE_std=MSE_std_mcmc, runtime=runtime_mcmc)
