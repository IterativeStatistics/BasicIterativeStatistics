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
affiliations:
 - name: EDF R&D, Saclay
   index: 1
 - name: Univ. Grenoble Alpes, Inria, CNRS, Grenoble INP, LIG, France
   index: 2
date: 27 June 2023
bibliography: paper.bib

---

# Statement of need

Global sensitivity analysis is an important step for analyzing and validating numerical simulations. The classical approach consists in collecting outputs on disk from multiple-simulations and then computing statistics after completion. Even though supercomputers continually enable larger simulations, scientists are constrained by slow and limited disk I/O, which typically means that simulations should be run at a low resolution with a limited number of probes.

# Summary

This library provides a fault tolerant Python package, called `Iterative-stats`, that enables high resolution global sensitivity analysis at large scale via on-the-fly statistics updates and a controlled intermediate storage. `Iterative-stats` was designed to be used independently from computation parallelization frameworks, which enables high bandwidth applications such as the example used in [@Schouler2023]. It will also soon be released with [@Salome]. Beyond existing applications, this package is also a playground for more experimental integrations compared to more established packages such as [@OpenTURNS] (for instance, Sobol indices).


# About the implemented methods

The implemented methods are: mean, variance,  higher-order moments (skewness and kurtosis), extrema, covariance, threshold (i.e. count the number of threshold exceedances) and Sobol indices. In particular, the library contains corrected formula compared to the original implementation made in [@Melissa] and some improvements (e.g. second order sobol index). It also implements exploratory work on quantile calculation based on the work of [@iooss:hal-03191621].

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

### Installation

The package is distributed with PyPi and can be installed with:

```bash
pip install iterative-stats
```

# Acknowledgements

We thanks Bruno Raffin (INRIA), Michael Baudin (EDF) and Joseph Mure (EDF) for following this work.


# References
