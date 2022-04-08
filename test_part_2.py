import numpy as np


from numpy.random import multivariate_normal, randn # Probability distributions on vectors
from scipy.linalg.special_matrices import toeplitz
import matplotlib.pyplot as plt 
from src.part_2 import Method
from utils.problem import Problem



### Set the problem ####
def simu_linmodel(w, n, std=1., corr=0.5):
        """
        Simulation values obtained by a linear model with additive noise
        
        Parameters
        ----------
        w : np.ndarray, shape=(d,)
            The coefficients of the model
        
        n : int
            Sample size
        
        std : float, default=1.
            Standard-deviation of the noise

        corr : float, default=0.5
            Correlation of the feature matrix
        """    
        d = w.shape[0]
        cov = toeplitz(corr ** np.arange(0, d))
        X = multivariate_normal(np.zeros(d), cov, size=n)
        noise = std * randn(n)
        y = X.dot(w) + noise
        return X, y

d = 100
n = 1
idx = np.arange(d)
lbda = 1. / n ** (0.5)

# Fix random seed for reproducibility
np.random.seed(0)

# Ground truth coefficients of the model - Linear regression
w_model_truth = (-1)**idx * np.exp(-idx / 10.)

Xlin, ylin = simu_linmodel(w_model_truth, n, std=1., corr=0.1)

problem = Problem(Xlin, ylin,lbda)

w01 = np.zeros(d)

# Logit Problem
Xlog, ylog = simu_linmodel(w_model_truth, n, std=1., corr=0.7)
pblogreg = Problem(Xlog, ylog,lbda,loss='logit')


######## Apply method ###############

method = Method(X=Xlin, w0=w01, nb=n, lbda=lbda, epsilon=10e100)

#### Question 2.2 -- Several stepsize with subsampling
## You can comment the wk_user to see clearly how the norm of wk, wk_ls and wk_l are evoluating
# wk, nw, ep = method.subsampling_newton(problem, 0.0001, 0.5, 1, 0)
# wk_ls, nw, ep = method.subsampling_newton(problem, 0.0001, 0.5, 1, 1)
# wk_l, nw, ep = method.subsampling_newton(problem, 0.0001, 0.5, 1, 2)
# wk_user, nw, ep = method.subsampling_newton(problem, 0.0001, 0.5, 2, 4)

# epoch = [i for i in range(ep)]

# plt.plot(epoch, wk, 'r', label="Armijo rule") # plotting t, a separately 
# plt.plot(epoch, wk_ls, 'b', label="Compute with L_sk") # plotting t, a separately 
# plt.plot(epoch, wk_l, 'g', label= "Compute with L") # plotting t, a separately
# plt.plot(epoch, wk_user, 'y', label= "Compute with alpha from user") # plotting t, a separately
# plt.xlabel("Epochs")
# plt.ylabel("Norm")
# plt.legend()
# plt.show()


### Question 2.3 -BFGS with different stepsoze
wk_ls,nw, ep = method.bgfs_subsampling(pblogreg, 0.0001, 0.5, 1, 1)
wk_l, nw, ep = method.bgfs_subsampling(pblogreg, 0.0001, 0.5, 1, 2)

epoch = [i for i in range(ep)]


# plt.plot(epoch, wk_ls, 'b', label="Compute with L_sk") # plotting t, a separately 
# plt.plot(epoch, wk_l, 'g', label= "Compute with L") # plotting t, a separately
# plt.xlabel("Epochs")
# plt.ylabel("Norm")
# plt.legend()
# plt.show()




## Regression problem -- Converging issue
# wk, nw, ep = method.bgfs_subsampling(problem, 0.0001, 0.5, 1, 1)
# epoch = [i for i in range(ep)]

# plt.plot(epoch, wk)
# plt.xlabel("Epochs")
# plt.ylabel("Norm")
# plt.legend()
# plt.show()
