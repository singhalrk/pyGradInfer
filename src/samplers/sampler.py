import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Sampler:
    def __init__(self,):
        pass

    def __call__(self, nSamples, nBurnin,
                 nChain, processes,
                 initX0):
        raise NotImplementedError

    def step(self, ):
        raise NotImplementedError
