# pyGradInfer


This is a python implementation of [deGradInfer](https://cran.r-project.org/web/packages/deGradInfer).
This is performing parameter estimation in non-linear ordinary differential
equation models, where we are aware of the system of equations but not aware
of the parameters, and the task is to infer the parameters and the states given
noisy and/or incomplete observations.


Imagine having observations of an apple falling on the moon, and where
we know the system of equations, this would module would give you the
gravitational constant.

These kind of problems are popular in systems biology where we have some prior
knowledge on the system of equations and in some cases we may have some
knowledge of the parameters or in some cases we may not be able to observe all
the states. For this reason we use a technique called Gaussian Process Gradient
Matching.
