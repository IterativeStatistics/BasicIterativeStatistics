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
        
       
        self.dotproduct_AE = [IterativeDotProduct(vector_size) for _ in range(self.nb_parms)]
        self.dotproduct_BE = [IterativeDotProduct(vector_size) for _ in range(self.nb_parms)]

        self.mean_tot = IterativeMean(vector_size)
        self.mean_A = IterativeShiftedMean(vector_size)
        self.mean_B = IterativeShiftedMean(vector_size)
        self.mean_E = [IterativeShiftedMean(vector_size) for _ in range(self.nb_parms)]
       
        

    def increment(self, data):
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]

        # update iteration, var
        self.iteration += 1
        self._increment_variance(data)

        # update all the means
        for d in data :
            self.mean_tot.increment(d)
        self.mean_A.increment(sample_A, shift=self.mean_tot.get_stats())
        self.mean_B.increment(sample_B, shift=self.mean_tot.get_stats())


        for p in range(self.nb_parms):
            self.mean_E[p].increment(sample_E[p], shift=self.mean_tot.get_stats())
            # update dot product
            self.dotproduct_BE[p].increment(sample_B,sample_E[p], shift = self.mean_tot.get_stats())
            self.dotproduct_AE[p].increment(sample_A,sample_E[p], shift = self.mean_tot.get_stats())
            

    def _compute_varianceI(self):
        return [self.dotproduct_BE[p].get_stats() - self.mean_A.get_stats() * self.mean_B.get_stats()[0] for p in range(self.nb_parms)]

    def _compute_VTi(self):
        
        return [self.var_A.get_stats()[0] - self.dotproduct_AE[p].get_stats() +  self.mean_A.get_stats() * self.mean_A.get_stats() for p in range(self.nb_parms)]
