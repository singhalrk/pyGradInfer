"""abstract sampler class"""
from diagnostics import stein, rubin_gelman
from abc import ABCMeta, abstractmethod


class Sampler(metaclass=ABCMeta):
    def __init__(self,):
        pass

    def __call__(self, nSamples, nBurnin,
                 nChain, processes,
                 initX0):
        raise NotImplementedError()

    @abstractmethod
    def step(self, ):
        raise NotImplementedError()

    def trace_plots(self,):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def diagnose(self):
        pass

    def r_statistic(self, samples, logp, d_logp):
        stein_divergence = stein(samples, logp, d_logp)
        rubin_stat = rubin_gelman(samples)

        return stein_divergence, rubin_stat
