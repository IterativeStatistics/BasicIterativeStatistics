# BasicIterativeStatistics
In this repository, basic iterative statistics are implemented.

## Installation

The python package **iterative_stats** is available on [pypi](https://pypi.org/project/iterative-stats/).
```
 pip install iterative-stats
```

Otherwise, one can clone the following repository:
```
    git clone https://github.com/IterativeStatistics/BasicIterativeStatistics.git
```

To install the environnement, please use poetry:

```
    poetry install
```

NB: One can also use a conda or python environment. The list of dependencies are available into the [pyproject.toml](pyproject.toml) file.

To run the tests:
```
    poetry run pytest tests
```

or for a specific test (ex: tests/unit/test_IterativeMean.py)

```
    poetry run pytest tests/unit/test_IterativeMean.py
```

## License

The python package **iterative_stats** free software distributed under the BSD 3-Clause License. The terms of the BSD 3-Clause License can be found in the file LICENSE.

## Iterative statistics

In this repository, we implement the following basic statistics:
- Mean (see example [here](./tests/unit/test_IterativeMean.py))
- Variance (see example [here](./tests/unit/test_IterativeVariance.py))
- Extrema (see example [here](./tests/unit/test_IterativeExtrema.py))
- Covariance (see example [here](./tests/unit/test_IterativeCovariance.py))

It also contains more advanced statistics: the Sobol indices. For each method (Martinez, Saltelli and Jansen), the first order indices, computed iteratively, as well as the total orders are available. We also include the second ordre for the Martinez and Jansen methods (the second order for the Saltelli method is still a work in progress).
- Pearson coefficient (Martinez): examples are available [here](tests/unit/sensitivity/test_IterativeSensitivityMartinez.py).
- Jansen method: examples are available [here](tests/unit/sensitivity/test_IterativeSensitivityJansen.py).
- Saltelli method: examples are available [here](tests/unit/sensitivity/test_IterativeSensitivityJansen.py).

NB: This package contains also useful methods for performing iterative statistics computations such as shift averaging and shift dot product computation:
- Shifted dot product (see example [here](./tests/unit/test_IterativeDotProduct.py))
- Shifted mean (see example [here](./tests/unit/test_IterativeMean.py))


### Examples

Here are some examples of how to use **iterative-stats** to compute Sobol index iteratively.

```python
from iterative_stats.sensitivity.sensitivity_martinez import IterativeSensitivityMartinez as IterativeSensitivityMethod
dim = 1 #field size
nb_parms = 3 #number of parameters
second_order = True # a boolean to compute the second order or not

# Create an instance of the object IterativeSensitivityMethod
sensitivity_instance = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = second_order)

# Generate an experimental design
from tests.mock.uniform_3d import Uniform3D
input_sample_generator = Uniform3D(nb_parms = nb_parms, nb_sim = nb_sim, second_order=second_order).generator()

# Load a function (here ishigami function)
from tests.mock.ishigami import ishigami
while True :
    try :
        # Generate the next sample
        input_sample = next(input_sample_generator)
        # Apply ishigami function
        output_sample = np.apply_along_axis(ishigami, 1,input_sample)
        # Update the sensitivity instance
        sensitivity_instance.increment(output_sample) 
    except StopIteration :
        break

first_order = sensitivity_instance.getFirstOrderIndices()
print(f" First Order Sobol indices (Martinez method): {first_order}")

total_order = sensitivity_instance.getFirstOrderIndices()
print(f" Total Order Sobol indices (Martinez method): {first_order}")

second_order = sensitivity_instance.getFirstOrderIndices()
print(f" Second Order Sobol indices (Martinez method): {first_order}")
```

NB: The computation of Sobol Indices requires the preparation of a specific experimental design based on the pick-freeze method (see [[1]](#1) for details). This method has been implemented into the class [**AbstractExperiment**](experimental_design/experiment.py) and some examples can be found [here](tests/unit/experimental_design/test_experiments.py).


## References 
The implementation of the iterative formulas is based on the following papers:

<a id="1">[1]</a>  Théophile Terraz, Alejandro Ribes, Yvan Fournier, Bertrand Iooss, and Bruno Raffin. 2017. Melissa: large scale in transit sensitivity analysis avoiding intermediate files. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '17). Association for Computing Machinery, New York, NY, USA, Article 61, 1–14. https://doi.org/10.1145/3126908.3126922

<a id="2">[2]</a> M. Baudin, K. Boumhaout, T. Delage, B. Iooss, and J-M. Martinez. 2016. Numerical stability of Sobol' indices estimation formula. In Proceedings of the 8th International Conference on Sensitivity Analysis of Model Output (SAMO 2016). Le Tampon, Réunion Island, France.

<a id="3">[3]</a> Philippe Pébay. 2008. Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. Sandia Report SAND2008-6212, Sandia National Laboratories 94 (2008).

