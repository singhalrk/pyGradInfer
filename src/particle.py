import numpy as np
from scipy.stats import rv_discrete

import ode_solvers as solvers


def initialize():
    raise NotImplementedError


def propOperator():
    raise NotImplementedError


def propogation(Xprev, Theta, Wn, eps,
                propOperator, shrinkage, evol_params):

    # number of particles
    nParticles = Xprev.shape[0]

    # average Theta with Wn
    averageTheta = (Wn * Theta).sum(axis=0)

    assert averageTheta.shape[1] == Theta.shape[1] == nParticles

    # shrink parameters
    newTheta = shrinkage*Theta + (1 - shrinkage)*averageTheta

    # get next step
    Xnext = propOperator(Xprev, newTheta, evol_params)

    return Xnext


def survival(Xn, Theta, Wn, eps, posterior):

    nParticles = Xn.shape[0]

    # fitness weights
    Gn = Wn * posterior(Xn, Theta)
    Gn /= Gn.sum(axis=0)

    # sampling new indices
    sampled_indices = rv_discrete(values=(np.arange(nParticles),
                                          Gn)).rvs(nParticles)

    Xnew = Xn[sampled_indices, :]
    newTheta = Theta[sampled_indices, :]

    return Xnew, newTheta


def proliferation(Xn, Theta, Wn, eps, shrinkage,
                  covFunction, propOperator, errorControl):
    # covariance matrix for Theta particles
    Cn = covFunction(Wn, Theta)

    # Sample new theta
    newTheta = Theta.copy() + np.random.multivariate_normal(mean=0,
                                                            cov=Cn,
                                                            size=Theta.shape)
    raise NotImplementedError


def weightUpdating():
    raise NotImplementedError
