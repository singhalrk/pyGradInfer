import ode_solvers as solvers
from weinberger_params import Weinberger_Params as params
import weinberger_params as Xinit

import numpy as np
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

    def feed_step(self, feed_parameters, x, x0, theta=None, eps=1e-3):
        """
        simulates a feed step

        feed_vol = volume to be removed and added
        feed_freq = number of times in a DAY we feed
        """

        # load feed terms
        feed_vol, _ = feed_parameters

        # take 1 step
        x = solvers.euler(x, self.f, eps=eps, params=theta)

        # load states
        dU, dI, dV, ds = x

        # load initial states
        U0, I0, V0, s0 = self.X0

        # volume to be removed
        dU += feed_vol * (U0 - dU)
        dI -= feed_vol * dI
        dV -= feed_vol * dI
        ds += feed_vol * (s0 - ds)

        return np.array([dU,
                         dI,
                         dV,
                         ds])

    def f_nonAutonomous(self, x, theta, t,
                        feed_parameters, eps, T, x0):

        # load feed parameters
        feed_vol, feed_freq = feed_parameters

        # load parameters
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

        feed_steps = np.arange(0, T, 1/(feed_freq + 1))[1:] / eps
        feed_steps = feed_steps.astype(int)

        if t in feed_steps:

            # initial conditions
            U0, _, _, s0 = self.X0

            # volume to be removed
            dU += feed_vol * (U0 - dU) / eps
            dI -= feed_vol * dI / eps
            dV -= feed_vol * dI / eps
            ds += feed_vol * (s0 - ds) / eps

        return np.array([dU,
                         dI,
                         dV,
                         ds])

    def get_trajectory(self, feed_parameters=[0.04, 1],
                       x0=None, T=2, theta=None,
                       eps=1e-3, prod=None):
        """use a delta function as feed function"""

        # feeding terms
        feed_vol, feed_freq = feed_parameters

        # initial conditions
        if x0 is None:
            x0 = self.X0

        f = self.f
        f_feed = self.feed_step

        # ode params
        if theta is None:
            theta = self.theta

            if prod is not None:
                theta[2] *= prod  # note this makes changes to self.theta

        # number of steps
        steps = int(T / eps)

        def f_time(x, theta, time):
            return self.f_nonAutonomous(x, theta, time,
                                        feed_parameters=feed_parameters,
                                        x0=x0, eps=eps, T=T)

        feed_steps = np.arange(0, T, 1/(feed_freq + 1))[1:] / eps
        feed_steps = feed_steps.astype(int)

        path = [x0.copy()]

        xCurr = x0.copy()

        for t in range(steps - 1):

            if t in feed_steps:
                xCurr = f_feed(feed_parameters, xCurr, x0, theta, eps)
                path.append(xCurr)
                continue

            xCurr = solvers.euler(xCurr, f, eps=eps, params=theta)
            path.append(xCurr)

        path = np.array(path)
        time = np.linspace(0, int(T), steps)

        self.f_nonAutonomous(x0, theta, t, feed_parameters, eps, T, x0)

        return path, time

    def get_observations(self, theta=None, x0=None, T=2,
                         feed_parameters=None,
                         eps=1e-3, sigma=5e+4, nObs=10,
                         prod=None):
        """use dimensionless equations to model stuff"""

        x, t = self.get_trajectory(x0=x0, T=T, eps=eps,
                                   feed_parameters=feed_parameters,
                                   theta=theta, prod=prod)

        if sigma is None:
            sigma = self.sigma

        x_noisy = x.copy()
        x_noisy += np.random.RandomState().normal(size=x.shape) * sigma

        N = int(T / (eps * nObs))

        x_noisy = x_noisy[::N]
        t = t[::N]

        """save experiment"""
        np.savetxt(f'data/x_vd_{prod}.txt', x_noisy)
        np.savetxt(f'data/t_vd_{prod}.txt', t)

        return x_noisy, t, x


if __name__ == '__main__':
    vd = ViralDynamics()

    feed_parameters = [0.10, 2]
    T = 1
    eps = 1e-3

    prod = 1

    path, _, x = vd.get_observations(feed_parameters=feed_parameters,
                                     T=T, prod=prod, eps=eps)
    U_t, I_t, V_t, s_t = x.T

    # feedless trajectory
    path2, t_noisy, x2 = vd.get_observations(feed_parameters=[0, 2],
                                             T=T, prod=prod,
                                             eps=eps)
    Ux, Ix, _, _ = x2.T

    t = np.arange(0, T, eps)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))

    ax[1].plot(t, U_t, label='feed U')
    ax[1].plot(t, I_t, label='feed I')

    ax[0].plot(t, Ux, label='no feed U')
    ax[0].plot(t, Ix, label='no feed I')

    ax[2].plot(t, I_t / (I_t + U_t), label='prevelance')
    ax[2].plot(t, Ix / (Ix + Ux), label='no feed prevelance')

    ax[3].plot(t, (I_t + U_t), label='vol')
    ax[3].plot(t, np.ones_like(U_t) * 1e+5)

    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    plt.show()
    plt.close()
