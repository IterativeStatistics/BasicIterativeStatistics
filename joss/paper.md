---
title: 'Iterative-stats: A Python package for basic iterative statistics'
tags:
  - Python
  - iterative statistics
authors:
  - name: Frédérique Robin
    orcid: 0009-0005-2141-7168
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Alejandro Ribes
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Robert Caulk
    equal-contrib: false
    affiliation: "2" # (Multiple affiliations must be quoted)
  - name: Marc Schouler
    equal-contrib: false
    affiliation: "2" # (Multiple affiliations must be quoted)
affiliations:
 - name: EDF R&D, Saclay
   index: 1
 - name: INRIA, Grenobles
   index: 2
date: 27 June 2023
bibliography: paper.bib

---

# Summary

Global sensitivity analysis is an important step for analyzing and validating numerical simulations. One classical approach consists in computing statistics on the outputs from well-chosen multiple simulation runs. Simulation results are stored to disk and statistics are computed postmortem. Even if supercomputers enable to run large studies, scientists are constrained to run low resolution simulations with a limited number of probes to keep the amount of intermediate storage manageable. In this work we propose a fault tolerant Python package that enables high resolution global sensitivity analysis at large scale. Statistics can thus be updated on-the-fly as soon as the in transit parallel server receives results from one of the running simulations.

# Statement of need
`Iterative-stats` is a Python package that compute standard statistics (e.g. mean, variance, etc.) on-the-fly with a controled intermediate storage.

`Iterative-stats` was designed to be used independently from computation parallelization frameworks. It has already been used in [@Melissa] and will also soon be included in [@Salome]. This package is also a playground for more robust integrations in packages such as [@OpenTURNS] (for instance, Sobol indices).


# About the implemented methods

The implemented method are : mean, variance,  higher-order moments (skewness and kurtosis), extrema, covariance, threshold (i.e. count the number of threshold exceedances) and Sobol indices. It also implements exploratory work on quantile calculation based on the work of [@iooss:hal-03191621].

Specifically, the iterative higher-order moments are based on [@Meng] work. As far as Sobol indices, three methods are implemented: Pearson coefficient (Martinez [@Martinez], Saltelli and Jansen) for the first and total order indices. These implimentation was validated by comparing it against the (non-iterative) statistics implemented in [@OpenTURNS]. We also implement the second order Sobol indices (for solely Pearson coefficient and Jansen methods).

This package contains also additional methods for performing iterative statistics computations such as shift averaging and shift dot product computation.

All the implemented methods are tested and are compared to non-iterative formulas (hard-coded or using OpenTURNS). Examples of implementation can be found into the [test repository](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests): 
- Mean (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeMean.py))
- Variance (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeVariance.py))
- Higher-order moments, skewness and kurtosis (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeMoments.py))
- Extrema (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeExtrema.py))
- Covariance (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeCovariance.py))
- Threshold (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeThreshold.py)) (count the number of threshold exceedances).
- Quantile (see examples [here](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/test_IterativeQuantile.py)): this statistics is still a work in progress and must be use with care!
- Sobol indices (see examples into the [sensitivity folder](https://github.com/IterativeStatistics/BasicIterativeStatistics/tree/main/tests/unit/sensitivity)) 

## Examples

### IterativeMean
Here is an example of how to compute mean iteratively :
```python
from iterative_stats.iterative_mean import IterativeMean
iterativeMean = IterativeMean(dim=1, state=state)

# Add a new state
for elt in [0.1, 0.2, 0.3] :
  iterativeMean.increment(elt)

# Get the computed iterative stats
print(f'The iterative mean is {iterativeMean.get_stats()}') 
```

### Fault-Tolerance 
For each statistics class, we implement a **save_state()** and **load_from_state()** methods to respectively save the current state and create a new object of type IterativeStatistics from a state object (a python dictionary).

These methods can be used as follows (as example):
```python
iterativeMean = IterativeMean(dim=1, state=state)

# ... Do some computations

# Save the current state
state_obj = iterativeMean.save_state() 

# Reload an IterativeMean object of state state_obj
iterativeMean_reload = IterativeMean(dim=1, state=state_obj)
```

# Acknowledgements

We thanks Bruno Raffin (INRIA), Michael Baudin (EDF) and Joseph Mure (EDF) for following this work.


# References
