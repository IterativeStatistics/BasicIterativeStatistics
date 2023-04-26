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
    def __init__(self, nb_parms: int, dim: int = 1, second_order: bool = False):
        super().__init__(nb_parms = nb_parms, dim = dim, second_order=second_order)
        self.var_B = IterativeVariance(dim)
        self.var_E = [IterativeVariance(dim) for _ in range(self.nb_parms)]
       
        self.covData_AE = [IterativeCovariance(dim) for _ in range(self.nb_parms)]
        self.covData_BE = [IterativeCovariance(dim) for _ in range(self.nb_parms)]
       
        self.pearson_A = np.zeros((dim, self.nb_parms))
        self.pearson_B = np.zeros((dim, self.nb_parms))
        del self.state 

       
    def _increment(self, data):
        sample_A = data[0]
        sample_B = data[1]
        sample_E = data[2:(2 + self.nb_parms)]
        self.var_B.increment(sample_B)

        for p in range(self.nb_parms):
            self.var_E[p].increment(sample_E[p])

            # update first order
            self.covData_BE[p].increment(sample_B,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_B.get_stats())
            self.pearson_B[:,p] = np.divide(self.covData_BE[p].get_stats(), np.sqrt(var_prod))

            # update last order
            self.covData_AE[p].increment(sample_A,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_A.get_stats())
            if var_prod.any() > 0 :
                self.pearson_A[:,p] = np.divide(self.covData_AE[p].get_stats(), np.sqrt(var_prod))
           

    def getFirstOrderIndices(self) :
        return self.pearson_B

    def getTotalOrderIndices(self) :
        return 1. - self.pearson_A

    def _compute_varianceI(self):
        res = np.zeros((self.dimension, self.nb_parms))
        for p in range(self.nb_parms):
            res[:,p] = np.multiply(self.pearson_B[:,p],self.var_A.get_stats())
        return res

    def _compute_VTi(self):
        return None
