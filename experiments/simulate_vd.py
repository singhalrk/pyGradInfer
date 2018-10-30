import ode_solvers as solvers
from weinberger_params import Weinberger_Params as params
import weinberger_params as Xinit

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class ViralDynamics:

    def __init__(self, sigma=1e-2):

        # load initial conditions
        U, I, V, s = Xinit.U0, Xinit.I0, Xinit.V0, Xinit.s0
        self.X0 = np.array([U, I, V, s])

        # load parameters
        self.parameters = params
        self.theta = list(self.parameters)

        # load noise term
        self.sigma = sigma

        # number of parameters
        self.nParams = len(self.theta)

    def f(self, x, theta):
        b_max = theta[0]
        K = theta[1]
        k = theta[2]
        d = theta[3]
        delta = theta[4]
        p = theta[5]
        c = theta[6]
        q_max = theta[7]

        U, I, V, s = x

        cap = s / (K + s)

        dU = b_max*cap*U - k*U*V - d*U
        dI = k*U*V - delta*I
        dV = p*I - c*V
        ds = -q_max*cap*(U + I)

        return np.array([dU,
                         dI,
                         dV,
                         ds])

    def get_trajectory(self,
                       x0=None, T=2, theta=None,
                       eps=1e-3, prod=None):
        """use a delta function as feed function"""

        # initial conditions
        if x0 is None:
            x0 = self.X0

        f = self.f

        # ode params
        if theta is None:
            theta = self.theta

            if prod is not None:
                theta[2] *= prod  # note this makes changes to self.theta

        # number of steps
        steps = int(T / eps)

        path = [x0.copy()]

        xCurr = x0.copy()

        for t in range(steps - 1):

            xCurr = solvers.euler(xCurr, f, eps=eps, params=theta)
            path.append(xCurr)

        path = np.array(path)
        time = np.linspace(0, int(T), steps)

        return path, time

    def get_observations(self, theta=None, x0=None, T=2,
                         eps=1e-3, sigma=5e+4, nObs=10,
                         prod=None):
        """use dimensionless equations to model stuff"""

        x, t = self.get_trajectory(x0=x0, T=T, eps=eps,
                                   theta=theta, prod=prod)

        if sigma is None:
            sigma = self.sigma

        x_noisy = x.copy()
        x_noisy += np.random.RandomState().normal(size=x.shape) * sigma

        N = int(T / (eps * nObs))

        x_noisy = x_noisy[::N]
        t = t[::N]

        """save experiment"""
        os.makedirs("data", exist_ok=True)
        np.savetxt("data/x_vd.txt", x_noisy)
        np.savetxt("data/t_vd.txt", t)

        return x_noisy, t, x


if __name__ == '__main__':
    vd = ViralDynamics()

    T = 1
    eps = 1e-3

    prod = 1

    path, _, x = vd.get_observations(T=T, prod=prod, eps=eps)
    U, I, V, s = x.T

    t = np.arange(0, T, eps)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))

    ax[0, 0].plot(t, U, label='U')
    ax[0, 1].plot(t, I, label='I')
    ax[1, 0].plot(t, V, label='V')
    ax[1, 1].plot(t, s, label='s')

    ax[0, 0].legend(loc="upper right")
    ax[0, 1].legend(loc="upper right")
    ax[1, 0].legend(loc="upper right")
    ax[1, 1].legend(loc="upper right")

    plt.show()
    plt.close()
