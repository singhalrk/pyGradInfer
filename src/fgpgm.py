import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import multiprocessing as mp

from kernels.rbf import RBF
from kernels.sigmoid import Sigmoid

import ode_solvers as solvers
from mcmc import MCMCBounded
# from lotka_volterra import LotkaVolterra


# TODO list
"""
  - [ ] TODO write prior for theta

  - [x] use log-normal to sample for parameters
  - [ ] TODO figure out proposals and parameters for log-normal
  - [x] try fixing certain values instead of sampling (in mcmc.sample_Prop)
  - [ ] find good interpolation algorithm for V and s
  - [ ] find scipy SEED
  - [ ] use george for GP regression
"""



class FGPGM:
    def __init__(self,  Y_loc, time_loc,
                 sigma=None,
                 gamma=None,
                 nHiddenStates=None, A=None,
                 nParams=None,
                 param_names=None,
                 which_kernel='sigmoid',
                 kernel_params=None,
                 F=None,
                 fixed_indices=None,
                 fixed_values=None,
                 bound_indices=None,
                 bounds=None,
                 proposalStd=None):

        # load all states ; shape - [time * nStates]
        self.Y = np.loadtxt(Y_loc)

        self.Y_loc = Y_loc
        self.time_loc = time_loc

        # load observation noise estimate
        self.sigma = sigma      # observation noise
        self.gamma = gamma      # gradient mismatch noise

        # load time observations
        time = np.loadtxt(time_loc)
        self.time = time
        self.time_diff = (time.reshape(-1, 1) - time.reshape(1, -1)).T

        # fixed indices
        self.fixed_indices = fixed_indices
        self.fixed_values = fixed_values

        # acceptable bounds for parameters
        self.bound_indices = bound_indices
        self.bounds = bounds

        # get proposal widths
        self.proposalStd = proposalStd

        # number of unobserved states
        self.nHiddenStates = nHiddenStates
        self.A = A              # interpolation matrix

        # number of params
        self.nParams = nParams
        self.param_names = param_names

        self.kernel_params = kernel_params

        # number of variables
        self.nStates = self.Y.shape[1]

        # number of time points
        self.nObs = self.time.size

        # load system of equations
        self.F = F

        # get RBF Kernel
        if kernel_params and which_kernel == 'rbf':
            self.kernels = [RBF(k_param[0],
                                k_param[1]) for k_param in kernel_params]

            # find optimal kernel params
            # self.fitGP()

            # load kernel and density related matrices
            self.get_Matrices()

            print('All RBF Matrices Loaded')

        # get Sigmoid Kernel
        if kernel_params and which_kernel == 'sigmoid':
            self.kernels = [Sigmoid(k_param) for k_param in kernel_params]

            # find optimal kernel params
            # self.fitGP()

            # load all kernel related matrices
            self.get_Matrices()

            print('All Sigmoid Matrices Loaded')

        # get sampler
        self.rnd = np.random.RandomState()
        self.MHsampler = MCMCBounded()

    def fitGP(self):
        """infers optimal kernel parameters for given data"""
        Y = self.Y
        time = self.time
        kernels = self.kernels

        # find optimal kernel params for each state
        # so we learn phi, sigma

        raise NotImplementedError

    def get_results(self, samples):
        nSamples = len(samples)
        nParams = self.nParams

        nStates = self.nStates
        nObs = self.nObs

        theta = np.zeros(nParams)
        X = np.zeros((nObs, nStates))

        theta_samples = []
        X_samples = []

        for sample in samples:
            theta += sample[:nParams]
            theta_samples.append(sample[:nParams])

            u, i, v, s = sample[nParams:]

            X.T[0] += u
            X.T[1] += i
            X.T[2] += v
            X.T[3] += s

            X_samples.append([u, i, v, s])

        return theta / nSamples, X / nSamples, theta_samples, X_samples

    def trace_plots(self, theta_samples, trace_name):
        nParams = self.nParams

        if self.fixed_indices is not None:
            fixed_indices = self.fixed_indices
            bound_indices = self.bound_indices
            bounds = self.bounds

            nFixed = len(fixed_indices)

        fig, ax = plt.subplots(nrows=2,
                               ncols=nParams - nFixed,
                               figsize=(14, 5))

        theta_samples = np.asarray(theta_samples)
        param_names = 'b_max,K,k,d,delta,p,c,q_max'.split(',')

        theta_samples = theta_samples.T
        theta_samples = theta_samples[bound_indices]

        for i, param in enumerate(theta_samples):

            ax[0, i].set_title(param_names[bound_indices[i]])

            sns.distplot(param, ax=ax[0, i])
            ax[1, i].plot(param)

        plt.savefig(f'plots/trace_{trace_name}')
        # plt.show()

    def get_Matrices(self):
        A_matrices = []
        m_moX_matrices = []
        Gamma_matrices = []
        Cphi_matrices = []

        dim = self.nStates
        kernels = self.kernels
        time = self.time
        nObs = len(time)
        t1 = time.reshape(-1, 1)
        t2 = time.reshape(1, -1)

        gamma = self.gamma

        for k in range(dim):
            kernel = kernels[k]
            gamma_k = gamma[k]

            DashC = kernel.DashC(t1, t2)
            inv_C = np.linalg.inv(kernel.k(t1, t2))
            CDash = kernel.CDash(t1, t2)
            CDoubleDash = kernel.CDoubleDash(t1, t2)

            m_noX_k = DashC.dot(inv_C)  # multiply with x_k later
            A_k = CDoubleDash - DashC.dot(inv_C.dot(CDash))

            Gamma_k = gamma_k*np.eye(nObs) + A_k

            A_matrices.append(A_k)
            m_moX_matrices.append(m_noX_k)
            Gamma_matrices.append(Gamma_k)

            Cphi_matrices.append(kernel.k(t1, t2))

        # store matrices
        self.A_matrices = A_matrices
        self.m_noX_matrices = m_moX_matrices
        self.Gamma_matrices = Gamma_matrices
        self.Cphi_matrices = Cphi_matrices

    @staticmethod
    def logN(x, mean, cov):
        """Double check if this is right"""
        return multivariate_normal.logpdf(x, mean=mean, cov=cov)

    def logPrior_params(self, Theta):
        logPrior_Theta = 0

        # uniform prior
        bounds = self.bounds

        for i, param in enumerate(Theta):
            lower_bound, upper_bound = bounds[i]
            pass

        return logPrior_Theta

    def logPosterior(self, arr):
        """return log posterior with all states being observed"""

        nParams = self.nParams

        nObs = self.nObs

        Theta, X = arr[:nParams], arr[nParams:]  # find another way
        Y = self.Y

        nStates = self.nStates

        logPrior_Theta = None      # TODO find right prior - log p(theta)
        logPrior_Theta = 0

        logPrior_X = 0             # GP Prior - sum log p(X_k | 0, Cphi_k)
        logObs = 0                 # sum log p(Y|X, sigma)
        logGradientMatch = 0       # sum of log p(f_k(x, theta) | m_k, Gamma_k)

        Gamma_matrices = self.Gamma_matrices
        CPhi_matrices = self.Cphi_matrices
        m_noX_matrices = self.m_noX_matrices

        sigma = self.sigma

        F = self.F
        F_val = F(X, Theta)

        for k in range(nStates):
            sigma_k = sigma[k]

            Gamma_k = Gamma_matrices[k]  # cov of x_k GP derivative
            CPhi_k = CPhi_matrices[k]    # cov of x_k GP Prior
            m_noX_k = m_noX_matrices[k]

            # x_k = X.T[k]
            x_k = X[k]

            m_k = m_noX_k.dot(x_k)  # mean of derivative of GP

            y_k = Y.T[k]

            # 1. GP prior - add log p(x_k | 0, Cphi_k)
            logPrior_X += self.logN(x=x_k,
                                    mean=np.zeros(nObs),
                                    cov=CPhi_k + np.eye(nObs)*1e-6)

            # 2. Observation - add log p(y_k | x_k, sigma*I)
            logObs += self.logN(x=y_k,
                                mean=x_k,
                                cov=np.eye(nObs) * sigma_k**2)

            # logObs += self.logN(x=y_k,
            #                     mean=x_k,
            #                     cov=np.eye(nObs) * sigma_k**2)

            # 3. gradient matching - add log p(f_k(X, theta) | m_k, Gamma_k)
            logGradientMatch += self.logN(F_val[k],  # check the shape
                                          mean=m_k,
                                          cov=Gamma_k)

        return logPrior_Theta + logPrior_X + logObs + logGradientMatch

    def get_Arg(self, X, Theta, arr=None):
        arr = []

        for param in Theta:
            arr.append(param)

        for x in X.T:
            arr.append(x)

        return arr

    def get_samples(self, x0, nSamples=1000, nBurnin=100):
        """implements mcmc with bounds"""

        # logP = self.logPosterior  # find right argument

        # load fixed values
        fixed_indices = self.fixed_indices
        fixed_values = self.fixed_values

        def logP_joint(x):
            return self.logPosterior(x)

        proposalStd = self.proposalStd
        bounds = self.bounds

        if proposalStd is None:
            proposalStd = []
            for x in x0:
                proposalStd.append(np.eye(x.size))

        if bounds is None:
            bounds = []

            for i, x in enumerate(x0):
                if i < 8:
                    bounds.append([0, 140])
                else:
                    bounds.append([0, x.max()*20])

        # get bounds for state variables
        samples = self.MHsampler.sampler(logP=logP_joint,
                                         proposalStd=proposalStd,
                                         bounds=bounds,
                                         x0=x0,
                                         nSamples=nSamples, nBurnin=nBurnin,
                                         fixed_indices=fixed_indices,
                                         fixed_values=fixed_values)

        return samples

    def inference(self, nSamples=100, nBurnin=100, theta0=None, trace_name=1):
        """infers X and theta given noisy observations"""

        nObs = self.nObs        # number of time observations
        nStates = self.nStates  # number of state variables

        nParams = self.nParams  # number of parameters

        F = self.F              # load function

        # for unobserved states
        nHidden = self.nHiddenStates  # number of unobserved states
        A = self.A                    # interpolation matrix

        # initialize theta0
        if theta0 is None:
            theta0 = np.random.randn(nParams)

        self.Y = np.loadtxt(self.Y_loc)
        Y = self.Y              # load observation

        # initialize X
        X0 = Y + np.random.randn(*Y.shape) * 1

        # get argument for sampler and posterior
        arr0 = self.get_Arg(X=X0, Theta=theta0)

        # get samples
        samples = self.get_samples(arr0,
                                   nSamples=nSamples,
                                   nBurnin=nBurnin)

        theta, X, theta_samples, X_samples = self.get_results(samples)

        theta_samples = np.array(theta_samples)

        return theta, X, theta_samples

    def test_result(self, theta, X_mean, true_Theta, x0, T=10, eps=1e-3,
                    sample_Theta=False, nSamples=2,
                    theta_samples=None, trace_name=None):

        raise NotImplementedError


def print_params(true_param, estim_param, pname, index):
    mean = estim_param.copy()
    for i in index:
        mean[i] = round(mean[i], 3)

    mean[2] = round(mean[2], 10)

    for p1, p2, name in zip(true_param, mean, pname):
        print(f'true {name} = {p1}, predicted {name} = {p2}')
