import numpy as np
import scipy as scp
from scipy.special import logit, expit
from scipy.stats import multivariate_normal
import pdb

# class logit_transform:
#     def __init__(self, low, high):
#         self.low = low
#         self.high = high    
#         # self.vmap_transform_to_R = np.vectorize(self.transform_to_R)  
#         # self.vmap_transform_backto_x = np.vectorize(self.transform_backto_x)
#         # self.vmap_jacobian_xtoR = np.vectorize(self.jacobian_xtoR)

#     def transform_to_R(self, x): 
#         frac = (x - self.low)/(self.high - self.low ) 
#         y = logit(frac)     
#         return y

#     def transform_backto_x(self,y):
#         x = self.low + (self.high - self.low) * expit(y) 
#         return x    
    
#     def jacobian_xtoR(self, y):
#         return (self.high - self.low) * expit(y) * (1-expit(y))

    
def transform_to_R(x, low, high):  
    frac = (x - low)/(high - low ) 
    y = logit(frac)     
    return y

def jacobian_xtoR(y, low, high):
    # print("y: ", y) 
    # print("expit(y): ", expit(y))
    # print("expit(y) * (1-expit(y)): ", expit(y) * (1-expit(y)))
    return (high - low) * expit(y) * (1-expit(y))

def transform_backto_x(y, low, high):   
    x = low + (high - low) * expit(y) 
    return x  



#vmap on the above functions
vmap_transform_to_R = np.vectorize(transform_to_R)  
vmap_transform_backto_x = np.vectorize(transform_backto_x)
vmap_jacobian_xtoR = np.vectorize(jacobian_xtoR)    



#####################################################################
#################################################### Bisection method ##################
def bisection_method(func, a, b, tolerance=1e-8, max_iterations=1000):
    """
    Bisection method for finding the root of a function.

    Parameters:
    - func: The function for which the root needs to be found.
    - a: The lower bound of the interval.
    - b: The upper bound of the interval.
    - tolerance: The desired accuracy of the root.
    - max_iterations: The maximum number of iterations allowed.

    Returns:
    - The approximate root of the function.
    """
    fa = func(a)
    fb = func(b)
    # print("fa, fb: ", fa, fb)
    # pdb.set_trace()
    if fa * fb >= 0:
        # raise ValueError("The function values at the interval endpoints must have opposite signs.")
        return max(a, b)

    last_value = None
    for i in range(max_iterations):
        c = (a + b) / 2
        if abs(func(c)) < tolerance:
            return c
        if func(c) * fa < 0:
            b = c
        else:
            a = c
        last_value = c

    if last_value is not None:
        return last_value

    raise ValueError("The method did not converge within the maximum number of iterations.")


#####################################################################
######################################## Zeroth order derivative ###############################
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

############## Another simplex to R #############
class SimplexToFromRd():
    def __init__(self, mu = 0, sig = .1):
        self.mu = mu
        self.sig = sig

    def simplextoR(self, f):
        n = len(f)
        log_f = np.log(f)
        t = log_f + (1/n) * (np.random.lognormal(mean=self.mu, sigma=self.sig, size=1) - sum(log_f))

        return t

    def simplexfromR(self, t):

        n = len(t)
        r = sum(t)
        exp_t = np.exp(t)
        f = exp_t / sum(exp_t)
        # Calculate logJacobian + log(L(r(t)))
        logJacobian = np.log(n) + sum(np.log(f)) + scp.stats.norm.logpdf(r, loc=self.mu, scale=self.sig)

        return f, logJacobian