# Summary

Developing and implementing different algorithmics to solve several simulation problems.

# Simulation Problems

## 1. Approximations of π with statistical analysis
 
An implementation of 2 different methods to obtain an estimation of the value of π:
1. Number of targets within a circle of radius unit thrown uniformly in the square surrounding it.
2. Throwing a needle of length L on a floor with slats of width D ([Buffon method](https://en.wikipedia.org/wiki/Buffon%27s_needle_problem)).

In both cases, a statistical analysis is performed with an analysis of the estimation error with respect to the ideal value, in comparison with the confidence interval obtained as a function of the number of launches, N. A t-Student test is performed to compare the differences in the errors and determine whether one of the estimators is more accurate.

## 2. Gaussian samples with statistical analysis

An implementation of an algorithm to generate samples of a Gaussian variable X of mean with the value of 1 and standard deviation unit with the value of 1, and independent samples of another variable, Y, generated with a Gaussian of the same parameters, but restricted to Y>0. Compared the means of X and Y with n samples, applying a t-Student test to compare their differences and determine the superior one. Compared the means with a symmetric and asymmetric test as a function of the number of samples, n.
