import numpy as np
from typing import Dict

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.iterative_covariance import IterativeCovariance
from iterative_stats.utils.logger import logger


class IterativeSensitivityMartinez(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Martinez formula.
    """
    def __init__(self, nb_parms: int, vector_size: int = 1):
        super().__init__(nb_parms = nb_parms, nb_sim = 1, vector_size = vector_size)
        self.var_B = IterativeVariance(vector_size)
        self.var_E = [IterativeVariance(vector_size) for _ in range(self.nb_parms)]
       
        self.covData_AE = [IterativeCovariance(vector_size) for _ in range(self.nb_parms)]
        self.covData_BE = [IterativeCovariance(vector_size) for _ in range(self.nb_parms)]
       
        self.state = {'pearson_A' : np.zeros(self.nb_parms), 'pearson_B' : np.zeros(self.nb_parms)}
       
    def increment(self, data):
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]

        # update iteration, var
        self.iteration += 1
        self._increment_variance(data)
        self.var_B.increment(sample_B)

        for p in range(self.nb_parms):
            self.var_E[p].increment(sample_E[p])

            # update first order
            self.covData_BE[p].increment(sample_B,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_B.get_stats())
            self.state['pearson_B'][p] = np.divide(self.covData_BE[p].get_stats(), np.sqrt(var_prod))

            # update last order
            self.covData_AE[p].increment(sample_A,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_A.get_stats())
            self.state['pearson_A'][p] = np.divide(self.covData_AE[p].get_stats(), np.sqrt(var_prod))
           

    def getFirstOrderIndices(self) :
        return self.state.get('pearson_B')

    def getTotalOrderIndices(self) :
        return 1 - self.state.get('pearson_A')

    def _compute_varianceI(self):
        return None

    def _compute_VTi(self):
        return None
