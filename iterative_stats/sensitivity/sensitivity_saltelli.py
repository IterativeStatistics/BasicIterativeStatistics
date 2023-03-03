import numpy as np
from typing import Dict

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.iterative_mean import IterativeMean, IterativeShiftedMean
from iterative_stats.utils.logger import logger


class IterativeSensitivitySaltelli(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Saltelli formula.
    """
    def __init__(self, nb_parms: int, vector_size: int = 1):
        super().__init__(nb_parms = nb_parms, nb_sim = 1, vector_size = vector_size)
        
        self.iterative_shifted_mean_A = IterativeShiftedMean(vector_size)
        self.iterative_shifted_mean_B = IterativeShiftedMean(vector_size)

        self.dotproduct_AE = [IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_A) for _ in range(self.nb_parms)]
        self.dotproduct_BE = [IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_B) for _ in range(self.nb_parms)]
        
        self.mean_tot = IterativeMean(vector_size)
        
       
    def increment(self, data):
        sample_A = data[:self.nb_sim][0]
        sample_B = data[self.nb_sim:2*self.nb_sim][0]
        sample_E = data[2*self.nb_sim:(2 + self.nb_parms)*self.nb_sim]

        # update iteration, var
        self.iteration += 1
        self._increment_variance(data)

        # update all the means
        for d in data :
            self.mean_tot.increment(d)

        for p in range(self.nb_parms):
            self.dotproduct_AE[p].increment(sample_A,sample_E[p], shift = self.mean_tot.get_stats())
            self.dotproduct_BE[p].increment(sample_B,sample_E[p], shift = self.mean_tot.get_stats())
        
        self.iterative_shifted_mean_A.increment(sample_A, shift = self.mean_tot.get_stats())
        self.iterative_shifted_mean_B.increment(sample_B, shift = self.mean_tot.get_stats())

        
    def _compute_varianceI(self):
        mean_A = self.iterative_shifted_mean_A.get_stats()
        mean_B = self.iterative_shifted_mean_B.get_stats()
        return [self.dotproduct_BE[p].get_stats() - mean_A * mean_B for p in range(self.nb_parms)]

    def _compute_VTi(self):
        mean_A = self.iterative_shifted_mean_A.get_stats()
        return [self.var_A.get_stats()[0] - self.dotproduct_AE[p].get_stats() +  mean_A * mean_A for p in range(self.nb_parms)]
