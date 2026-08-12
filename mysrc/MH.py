import numpy as np
from scipy.stats import uniform
import torch
from mysrc.utils import bisection_method
import torch.distributions.multivariate_normal as MVN
import torch.distributions.normal as normal
from scipy.stats import multivariate_normal


def MH(logdensityfunc, x0, sigma, nmoves=5, return_entire_chain=False, adapt=True, adapt_no = 100):
    acceptance = 0
    d = x0.shape[0]
    #print(sigma.dim())

    x0chainnumpy = [x0.detach().numpy()]
    if return_entire_chain:
        x0chain = [x0.view(-1)]

    for iter in range(nmoves):
        print('Fraction of steps:',iter/nmoves,'(Total:',nmoves,')')
        if np.remainder(iter, adapt_no)==0 and iter>0 and sigma.dim()>0 and adapt:
            sigma = torch.tensor((5.66/d)*(np.cov(np.array(x0chainnumpy)[-100:,:].T)+1e-10*np.eye(d)))
            #print('Time to update to sigma:', sigma)

        ### Log-Exp transformation used to return on real line as the parameter is positive valued ###
        if sigma.dim() == 0:
            x_new = x0 + normal.Normal(torch.tensor([0.0]), sigma * torch.tensor([1.0])).sample(sample_shape=torch.Size([1]))[0].to(dtype=torch.float32)
        else:
            x_new = x0 + MVN.MultivariateNormal(torch.zeros(d, dtype=torch.double), sigma).sample(sample_shape=torch.Size([1]))[0].to(dtype=torch.float32)
        alpha = np.log(uniform.rvs(0, 1, 1)[0])
        if alpha < min(logdensityfunc(x_new)-logdensityfunc(x0),0):
            xt = x_new
            #print('Accepted')
            #print(min(np.exp(logdensityfunc(x_new)-logdensityfunc(x0)),1), 'Accepted')
            acceptance = acceptance + 1
        else:
            xt = x0
            #print('Rejected')
        if torch.isnan(xt).any():
            print("nan values in the MCMC")
            break
        else:
            x0 = xt
        #print('Updated value', x0)
        #if return_entire_chain:

        print('Acceptance rate:', acceptance / (iter+1))

        x0chainnumpy.append(x0.detach().numpy())
        if return_entire_chain:
            x0chain.append(x0.view(-1))

    if return_entire_chain:
        return torch.stack(x0chain)
    else:
        return xt.view(-1)

def zeroth_order_grad(func, x, mu, b):
    n_param = x.shape[0]
    f_x = func(x)
    grad = np.zeros_like(x)
    for i in range(b):
        random_u = multivariate_normal.rvs(mean = np.zeros(n_param), cov = np.eye(n_param), size = 1)
        new_x = x + random_u * mu
        f_xplusu = func(new_x)
        new_grad = ((f_xplusu - f_x) * random_u)
        grad = grad + new_grad/(mu * b)
    return grad