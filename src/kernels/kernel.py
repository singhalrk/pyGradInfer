import numpy as np
import matplotlib.pyplot as plt


class Kernel:
    def __init__(self,):
        pass

    def __call__(self, t1, t2):
        raise NotImplementedError

    def C_phi(self, t1, t2):
        raise NotImplementedError

    def phi_C(self, t1, t2):
        raise NotImplementedError
