import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sampler import Sampler


class Metropolis(Sampler):
    def __call__(self, nSamples, nBurnin,
                 nChains, processes, initX0):
        raise NotImplementedError

    def step(self, ):
        raise NotImplementedError
