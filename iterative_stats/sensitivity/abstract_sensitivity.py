import numpy as np
from typing import Dict

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger



class IterativeAbstractSensitivity(AbstractIterativeStatistics):
    """
        Abstract class to compute the first and total sensitivity indices
    """

    def __init__(self, nb_parms : int, nb_sim : int, vector_size: int = 1) -> None:
        super().__init__(vector_size)
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.var_A: AbstractIterativeStatistics = IterativeVariance(vector_size)
 
    def _increment_variance(self, data : np.array) -> None:        
        self.var_A.increment(data[:self.nb_sim])
        
    def getIteration(self) -> int:
        return self.iteration

    def getFirstOrderIndices(self) -> np.array:
        """
            Compute the self.nb_parms first order sensitivity indices
        """
        if self.iteration > 1 :
            return self._compute_varianceI()/self.var_A.get_stats()[0]
        else :
            return None
 
    def getTotalOrderIndices(self) -> np.array :
        """
            Compute the self.nb_parms total order sensitivity indices
        """
        if self.iteration > 1 :
            return self._compute_VTi()/self.var_A.get_stats()[0]
        else :
            return None

    def increment(self, data : np.array) -> None :
        """
            Function that applies the iterative formula to increment the first and total order indices.
        """
        raise Exception('Not implemented method')



    def _compute_varianceI(self) -> None:
        """
            Function that computes the variance V_i
        """
        raise Exception('Not implemented method')

    def _compute_VTi(self) -> None:
        """
            Function that computes the total variance V[Y] - V_-i
        """
        raise Exception('Not implemented method')

