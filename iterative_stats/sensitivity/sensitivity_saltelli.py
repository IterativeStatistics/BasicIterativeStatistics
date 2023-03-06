import numpy as np
from typing import Dict

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.utils.logger import logger
from iterative_stats.sensitivity import SALTELLI

class IterativeSensitivitySaltelli(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Saltelli formula.
    """
    def __init__(self, nb_parms: int, vector_size: int = 1, second_order: bool = False):
        super().__init__(nb_parms = nb_parms, nb_sim = 1, vector_size = vector_size, 
                            second_order=second_order, name = SALTELLI)
        
        self.dotproduct_AE = [IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_A, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i]) for i in range(self.nb_parms)]
        self.dotproduct_BE = [IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_B, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i]) for i in range(self.nb_parms)]
        
       
    def _increment(self, data):
        sample_E = data[2*self.nb_sim:(2 + self.nb_parms)*self.nb_sim]
        for p in range(self.nb_parms):
            self.dotproduct_AE[p].increment(data[:self.nb_sim],sample_E[p], shift = self.mean_tot.get_stats())
            self.dotproduct_BE[p].increment(data[self.nb_sim:2*self.nb_sim],sample_E[p], shift = self.mean_tot.get_stats())
        
    def _compute_varianceI(self):
        mean_A = self.iterative_shifted_mean_A.get_stats()
        mean_B = self.iterative_shifted_mean_B.get_stats()
        return [self.dotproduct_BE[p].get_stats() - mean_A * mean_B for p in range(self.nb_parms)]

    def _compute_VTi(self):
        mean_A = self.iterative_shifted_mean_A.get_stats()
        return [self.var_A.get_stats()[0] - self.dotproduct_AE[p].get_stats() +  mean_A * mean_A for p in range(self.nb_parms)]
