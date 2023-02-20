import numpy as np
from typing import Dict

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.utils.logger import logger


class IterativeSensitivitySaltelli(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Saltelli formula.
    """
    def __init__(self, nb_parms: int, vector_size: int = 1):
        super().__init__(nb_parms = nb_parms, nb_sim = 1, vector_size = vector_size)
        self.mean_E = [IterativeMean(vector_size) for _ in range(self.nb_parms)]
       
        self.prod_AE = [IterativeDotProduct(vector_size) for _ in range(self.nb_parms)]
        self.prod_BE = [IterativeDotProduct(vector_size) for _ in range(self.nb_parms)]

        self.mean_tot = IterativeMean(vector_size)
        self.mean_A = IterativeCenteredMean(vector_size)
        self.mean_B = IterativeCenteredMean(vector_size)
       
        

    def increment(self, data):
        prev_mean = self.mean_tot.get_stats() 

        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]

        # update iteration, var
        self.iteration += 1
        self._increment_variance(data)

        # update all the means
        for d in data :
            self.mean_tot.increment(d)
        self.mean_A.increment(sample_A, external_mean=self.mean_tot.get_stats())
        self.mean_B.increment(sample_B, external_mean=self.mean_tot.get_stats())


        for p in range(self.nb_parms):

            # update covariance
            self.prod_BE[p].increment(sample_B - self.mean_tot.get_stats(),sample_E[p] - self.mean_tot.get_stats())
            self.prod_AE[p].increment(sample_A - self.mean_tot.get_stats(),sample_E[p] - self.mean_tot.get_stats())
            

    def _compute_varianceI(self):
        return [self.prod_BE[p].get_stats()[0]/(self.iteration - 1) - self.var_A.get_mean()[0] * self.mean_E[p].get_stats()[0] for p in range(self.nb_parms)]

    def _compute_VTi(self):
        mean_A = self.var_A.get_mean()[0]
        return [self.var_A.get_stats()[0] - self.covData_AE[p].get_stats()[0] +  mean_A * mean_A for p in range(self.nb_parms)]
