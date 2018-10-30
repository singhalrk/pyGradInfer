import numpy as np


def euler(x, f, eps, params=None):
    return x + eps * f(x, params)


def rk2(x, f, eps, params=None):
    """RK2 Heun"""

    k1 = f(x, params)
    k2 = f(x + eps*k1, params)
    return x + (k1 + k2)*(eps / 2)


def rk4(x, f, eps, params=None):
    k1 = f(x, params)
    k2 = f(x + 0.5 * eps * k1, params)
    k3 = f(x + 0.5 * eps * k2, params)
    k4 = f(x + eps * k3, params)

    return x + (eps / 6) * (k1 + 2*k2 + 2*k3 + k4)


def solve(x0, F, T, eps=1e-3, method_name='rk4', params=None):
    if method_name == 'rk4':
        method = rk4
    elif method_name == 'euler':
        method = euler

    x0 = np.asarray(x0)
    path = [x0]

    N = int(T / eps)            # time steps

    for i in range(N - 1):
        if params is None:
            x0 = method(x0, F, eps)
        else:
            x0 = method(x0, F, eps, params)

        path.append(x0)

    return np.array(path)
