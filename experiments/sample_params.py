from collections import namedtuple
import numpy as np


Parameters = namedtuple('Parameters', 'b_max, K, k, d, delta, p, c, q_max')

d = 0.02  # uninfected-cell death rate
delta = 0.7  # infected-cell death rate
c = 30   # virion decay rate
k = 1.875e-7  # infectivity rate

n = 200
p = n*delta  # virion production rate

U0 = 8e+5
I0 = 0
V0 = 1e+6
s0 = 8e+5 * 10

b_max = 40
q_max = 25
K = 2e+6

X0 = np.array([U0, I0, V0, s0])

Sample_Params = Parameters(d=d,
                           K=K,
                           k=k,
                           q_max=q_max,
                           b_max=b_max,
                           p=p,
                           c=c,
                           delta=delta)
