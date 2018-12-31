import os
from tqdm import tqdm
import argparse

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from Kernels.rbf import RBF
from lotkavolterra import LotkaVolterra


class SVI:
    def __init__(self,  Y_loc, time_loc,
                 sigma=None,
                 gamma=None,
                 nHiddenStates=None, A=None,
                 nParams=None,
                 kernel_params=None,
                 get_BTheta=None,
                 get_BX=None,
                 F=None):

        # load all states ; shape - [time * nStates]
        self.Y = np.loadtxt(Y_loc)

        # load observation noise estimate
        self.sigma = sigma

        self.gamma = gamma      # change this later

        # load time observations
        self.time = np.loadtxt(time_loc)
        self.time_diff = (self.time.reshape(-1, 1) - self.time.reshape(1, -1))
        self.time_diff = self.time_diff.T

        # number of unobserved states
        self.nHiddenStates = nHiddenStates
        self.A = A

        # number of params
        self.nParams = nParams

        self.kernel_params = kernel_params

        # number of variables
        self.nStates = self.Y.shape[1]

        # number of time points
        self.nObs = self.time.size

        # get kernel
        self.kernels = [RBF(k_param[0],
                            k_param[1]) for k_param in kernel_params]

        # load all kernel related matrices
        self.get_kernel_matrices()
        print('got all matrices')

        # get B_matrices functions
        self.get_BTheta_function = get_BTheta
        self.get_BX = get_BX

        # get sampler
        self.rnd = np.random.RandomState()

    def get_kernel_matrices(self, A=None):
        """
        returns the
        1. A_k, m_k (without x) matrices
        2. get Gamma_k matrix as well
        3. mu_k vector and Sigma_k matrices
        """
        # get all kernel related matrices in one go
        dim = self.nStates
        kernels = self.kernels
        time = self.time
        nObs = len(time)
        t1 = time.reshape(-1, 1)
        t2 = time.reshape(1, -1)
        sigma = self.sigma
        gamma = self.gamma
        Y = self.Y

        m_noX_matrices = []
        A_matrices = []
        mu_vectors = []
        Sigma_matrices = []
        inv_Sigma_matrices = []
        Gamma_matrices = []

        for k in range(dim):
            kernel = kernels[k]
            sigma_k, gamma_k = sigma[k], gamma[k]
            y_k = Y.T[k]

            DashC = kernel.DashC(t1, t2)
            inv_C = np.linalg.inv(kernel.k(t1, t2))
            CDash = kernel.CDash(t1, t2)
            CDoubleDash = kernel.CDoubleDash(t1, t2)

            m_noX_k = DashC.dot(inv_C)  # multiply with x_k later
            A_k = CDoubleDash - DashC.dot(inv_C.dot(CDash))

            Gamma_k = gamma_k*np.eye(nObs) + A_k

            inv_sigma_k = 1/(sigma_k**2)
            if A is None:
                inv_Sigma_k = inv_C + np.eye(nObs)*inv_sigma_k
                Sigma_k = np.linalg.inv(inv_Sigma_k)

                mu_k = Sigma_k.dot(y_k) * inv_sigma_k
            else:
                inv_Sigma_k = inv_sigma_k*A.T.dot(A) + inv_C
                Sigma_k = np.linalg.inv(Sigma_k)
                mu_k = Sigma_k.dot(A.T.dot(y_k)) * inv_sigma_k

            mu_vectors.append(mu_k)
            inv_Sigma_matrices.append(inv_Sigma_k)
            Sigma_matrices.append(Sigma_k)
            m_noX_matrices.append(m_noX_k)
            A_matrices.append(A_k)
            Gamma_matrices.append(Gamma_k)

        # store them
        self.m_noX_matrices = m_noX_matrices
        self.A_matrices = A_matrices
        self.inv_Sigma_matrices = inv_Sigma_matrices
        self.Sigma_matrices = Sigma_matrices
        self.mu_vectors = mu_vectors
        self.Gamma_matrices = Gamma_matrices

    def get_rTheta(self, X):
        dim = X.shape[1]        # number of state variables
        nObs = X.shape[0]       # number of time observations

        assert X.shape[1] == self.Y.shape[1]

        m_noX_matrices = self.m_noX_matrices
        mat_Theta = self.get_B_Theta_matrix(X=X)
        B_Theta, b_Theta = mat_Theta

        Gamma_matrices = self.Gamma_matrices

        nParams = self.nParams

        rTheta = np.zeros(nParams)  # verrify it
        inv_Omega_Theta = np.zeros((nParams, nParams))
        Omega_Theta = np.zeros((nParams, nParams))
        for k in range(dim):
            B_Theta_k, b_Theta_k = B_Theta[k], b_Theta[k]
            Gamma_k = Gamma_matrices[k]

            m_noX_k = m_noX_matrices[k]
            x_k = X.T[k]
            m_k = m_noX_k.dot(x_k)  # check dimensions

            inv_Omega_Theta += B_Theta_k.T.dot(Gamma_k.dot(B_Theta_k))
            rTheta += B_Theta_k.T.dot(Gamma_k.dot(m_k - b_Theta_k))

        Omega_Theta = np.linalg.inv(inv_Omega_Theta)
        rTheta = Omega_Theta.dot(rTheta)

        return rTheta, Omega_Theta, inv_Omega_Theta

    def get_rX(self, theta, X):
        dim = X.shape[1]        # number of state variables
        nObs = X.shape[0]       # number of time observations

        assert X.shape[1] == self.Y.shape[1]

        m_noX_matrices = self.m_noX_matrices
        inv_Sigma_matrices = self.inv_Sigma_matrices
        mu_vectors = self.mu_vectors

        mat_X = self.get_BX_matrix(X=X, theta=theta)
        B_X, b_X = mat_X

        Gamma_matrices = self.Gamma_matrices

        r_X, Omega_X, inv_Omega_X = [], [], []

        for u in range(dim):
            r_X_u = np.zeros(nObs)
            Omega_X_u = np.zeros((nObs, nObs))
            inv_Omega_X_u = np.zeros((nObs, nObs))

            inv_Sigma_u = inv_Sigma_matrices[u]
            mu_u = mu_vectors[u]

            for k in range(dim):
                B_uk, b_uk = B_X[u][k], b_X[u][k]
                Gamma_k = Gamma_matrices[k]

                x_k = X.T[k]
                m_k = m_noX_matrices[k].dot(x_k)

                inv_Omega_X_u += B_uk.T.dot(Gamma_k.dot(B_uk)) + inv_Sigma_u

                r_X_u += B_uk.T.dot(Gamma_k.dot(m_k - b_uk))
                r_X_u += inv_Sigma_u.dot(mu_u)

            Omega_X_u = np.linalg.inv(inv_Omega_X_u)
            r_X_u = Omega_X_u.dot(r_X_u)

            r_X.append(r_X_u)
            Omega_X.append(Omega_X_u)
            inv_Omega_X.append(inv_Omega_X_u)

        return r_X, Omega_X, inv_Omega_X

    def get_B_Theta_matrix(self, X):
        B_Theta, b_theta = self.get_BTheta_function(X)
        return B_Theta, b_theta

    def get_BX_matrix(self, X, theta):
        B_X, b_X = self.get_BX(X, theta)
        return [B_X, b_X]

    def optimize_kernel(self):
        """get from FGPGM"""
        raise NotImplementedError

    def sample_Theta(self, mu, cov, nSamples):

        return self.rnd.multivariate_normal(mean=mu,
                                            cov=cov,
                                            size=nSamples)

    def sample_x_k(self, mu, cov, nSamples):

        return self.rnd.multivariate_normal(mean=mu,
                                            cov=cov,
                                            size=nSamples)

    def sample_X(self, mu_vectors, cov_matrices, nSamples):
        '''sample entire X'''
        dim = self.nStates      # number of states
        nObs = self.nObs
        assert dim == len(mu_vectors)

        X_samples = np.zeros((nSamples, nObs, dim))

        for k in range(dim):
            mu_k, cov_k = mu_vectors[k], cov_matrices[k]
            x_k_samples = self.sample_x_k(mu_k, cov_k, nSamples)

            X_samples[:, :, k] = x_k_samples
        return X_samples

    def update_qTheta(self, samples):
        new_mean = np.zeros(self.nParams)
        new_cov = np.zeros((self.nParams, self.nParams))

        nSamples = samples.shape[0]

        for sample in samples:

            mu, cov, _ = self.get_rTheta(X=sample)

            new_mean += mu
            new_cov += cov

        return new_mean/nSamples, new_cov/nSamples

    def update_qX_k(self, samples, k):
        nObs = self.nObs

        X_samples, theta_samples = samples
        new_mean = np.zeros(nObs)
        new_cov = np.zeros((nObs, nObs))

        nSamples = X_samples.shape[0]

        for X, theta in zip(X_samples, theta_samples):
            mu, cov, _ = self.get_rX(theta=theta, X=X)

            new_mean += mu[k]
            new_cov += cov[k]

        return new_mean/nSamples, new_cov/nSamples

    def coordinate_ascent(self, epochs=100, nSamples=1000):
        """
        nSamples - number of x_i and theta samples to take
        to calculate parameters for mean field
        coordinate ascent
        """
        nObs = self.nObs        # time points
        sigma = self.sigma       # observation noise
        nParams = self.nParams
        dim = self.nStates

        Y = self.Y
        assert dim == Y.shape[1]        # number of state variables

        # initialize random variables
        X0 = Y + np.random.randn(*Y.shape)     # initialize X
        theta0 = np.random.randn(nParams) + 0  # initialize theta

        # initialize mean and Covariance matrices
        r_X, Omega_X, inv_Omega_x = self.get_rX(X=X0,
                                                theta=theta0)

        r_Theta, Omega_Theta, inv_Omega_Theta = self.get_rTheta(X=X0)

        for t in tqdm(range(epochs)):

            # 1. update qX distribution sequentially first
            for k in range(dim):

                r_X_k = r_X[k]
                Omega_X_k = Omega_X[k]

                theta_samples = self.sample_Theta(mu=r_Theta,
                                                  cov=Omega_Theta,
                                                  nSamples=nSamples)

                # check if this is correct
                X_samples = self.sample_X(r_X,
                                          Omega_X,
                                          nSamples)

                samples = [X_samples, theta_samples]
                r_X_k, Omega_X_k = self.update_qX_k(samples, k)

                # update r_X[k] and Omega_X[k] - var parameters for x_k
                r_X[k] = r_X_k.copy()
                Omega_X[k] = Omega_X_k.copy()

            # 2. update qTheta distribution
            X_Samples = self.sample_X(r_X, Omega_X,
                                      nSamples=nSamples)

            r_Theta, Omega_Theta = self.update_qTheta(samples=X_Samples)

        return r_Theta, Omega_Theta, r_X, Omega_X
