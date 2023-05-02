import numpy as np

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.utils.logger import logger
from iterative_stats.sensitivity import SALTELLI

class IterativeSensitivitySaltelli(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Saltelli formula.
    """
    def __init__(self, nb_parms: int, dim: int = 1, second_order: bool = False, state: object = None):
        super().__init__(nb_parms = nb_parms, dim = dim, 
                            second_order=second_order, name = SALTELLI, state=state)
        
        self.dotproduct_AE = [IterativeDotProduct(dim, iterative_shifted_mean_1 = self.iterative_shifted_mean_A, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i]) for i in range(self.nb_parms)]
        self.dotproduct_BE = [IterativeDotProduct(dim, iterative_shifted_mean_1 = self.iterative_shifted_mean_B, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i]) for i in range(self.nb_parms)]
        
       
    def _increment(self, data):
        sample_E = data[2:(2 + self.nb_parms)]
        for p in range(self.nb_parms):
            self.dotproduct_AE[p].increment(data[0],sample_E[p], shift = self.mean_tot.get_stats())
            self.dotproduct_BE[p].increment(data[1],sample_E[p], shift = self.mean_tot.get_stats())

        
    def _compute_varianceI(self):
        mean_A = self.iterative_shifted_mean_A.get_stats()
        mean_B = self.iterative_shifted_mean_B.get_stats()
        var = np.zeros((self.dimension, self.nb_parms))
        for p in range(self.nb_parms):
            var[:,p] = self.dotproduct_BE[p].get_stats() - mean_A * mean_B
        return var

    def _compute_VTi(self):
        vti = np.zeros((self.dimension, self.nb_parms))
        mean_A = self.iterative_shifted_mean_A.get_stats()
        for p in range(self.nb_parms):
            vti[:,p] = self.var_A.get_stats() - self.dotproduct_AE[p].get_stats() +  mean_A * mean_A
        return vti

    def getSecondOrderIndices(self) -> np.array:
        """
            Compute the self.nb_parms second order sensitivity indices
        """
        raise Exception('Not implemented method')

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()

        state['dotproduct_AE'] = [self.dotproduct_AE[i].save_state() for i in range(self.nb_parms)]
        state['dotproduct_BE'] = [self.dotproduct_BE[i].save_state() for i in range(self.nb_parms)]

        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)

        s = state.get('dotproduct_AE')
        self.dotproduct_AE = [IterativeDotProduct(self.dimension, iterative_shifted_mean_1 = self.iterative_shifted_mean_A, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i], state = s[i]) for i in range(self.nb_parms)]
        
        s = state.get('dotproduct_BE')
        self.dotproduct_BE = [IterativeDotProduct(self.dimension, iterative_shifted_mean_1 = self.iterative_shifted_mean_B, iterative_shifted_mean_2 = self.iterative_shifted_mean_E[i], state = s[i]) for i in range(self.nb_parms)]
        
