---
title: 'Iterative-stats: A Python package for basic iterative statistics'
tags:
  - Python
  - iterative statistics
authors:
  - name: Frédérique Robin, ...
    orcid: 0009-0005-2141-7168
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: EDF R&D, Saclay
   index: 1
date: 27 June 2023
bibliography: paper.bib

---

# Summary

Global sensitivity analysis is an important step for analyzing and validating numerical simulations. One classical approach consists in computing statistics on the outputs from well-chosen multiple simulation runs. Simulation results are stored to disk and statistics are computed postmortem. Even if supercomputers enable to run large studies, scientists are constrained to run low resolution simulations with a limited number of probes to keep the amount of intermediate storage manageable. In this work we propose a fault tolerant Python package that enables high resolution global sensitivity analysis at large scale. Statistics can thus be updated on-the-fly as soon as the in transit parallel server receives results from one of the running simulations.

# Statement of need
`Iterative-stats` is a Python package that compute standard statistics (e.g. mean, variance, etc.) on-the-fly with a controled intermediate storage.

`Iterative-stats` was designed to be used independently from computation parallelization frameworks. It has already been used in [@Melissa] and will also soon be included in [@Salome]. This package is also a playground for more robust integrations in packages such as [@OpenTURNS].


# About the implemented methods

The implemented method are : mean, variance,  higher-order moments (skewness and kurtosis), extrema, covariance, threshold (i.e. count the number of threshold exceedances) and Sobol indices. It also implements exploratory work on quantile calculation based on the work of [@iooss:hal-03191621].

Specifically, the iterative higher-order moments are based on [@Meng] work. As far as Sobol indices, three methods are implemented: Pearson coefficient (Martinez [@Martinez], Saltelli and Jansen) for the first and total order indices. These implimentation was validated by comparing it against the (non-iterative) statistics implemented in [@OpenTURNS]. We also implement the second order Sobol indices (for solely Pearson coefficient and Jansen methods).

This package contains also additional methods for performing iterative statistics computations such as shift averaging and shift dot product computation.

# Acknowledgements



# References