import numpy as np
from typing import Dict

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeShiftedMean, IterativeMean
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger



class IterativeAbstractSensitivity(AbstractIterativeStatistics):
    """
        Abstract class to compute the first and total sensitivity indices
    """

    def __init__(self, nb_parms : int, nb_sim : int, vector_size: int = 1, second_order: bool = False) -> None:
        super().__init__(vector_size)
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.var_A: AbstractIterativeStatistics = IterativeVariance(vector_size)
        self.second_order = second_order

        if self.second_order :
            self.mean_tot = IterativeMean(vector_size)
            self.iterative_shifted_mean_A = IterativeShiftedMean(vector_size)
            self.iterative_shifted_mean_E = [IterativeShiftedMean(vector_size) for _ in range(self.nb_parms)]
            self.iterative_shifted_mean_B = IterativeShiftedMean(vector_size)

            self.dotproduct_AB = IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_A, iterative_shifted_mean_2 = self.iterative_shifted_mean_B)
            self.dotproduct_EC = [[IterativeDotProduct(vector_size, iterative_shifted_mean_1 = self.iterative_shifted_mean_E[i]) for i in range(self.nb_parms)] for _ in range(self.nb_parms)]

 
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
 
    def getSecondOrderIndices(self) -> np.array:
        """
            Compute the self.nb_parms second order sensitivity indices
        """
        if self.iteration > 1 and self.second_order :   
            first_order = self._compute_varianceI()
            val = np.array([[self.dotproduct_EC[j][i].get_stats() - first_order[i] - first_order[j] for i in range(self.nb_parms)] for j in range(self.nb_parms)]) 
            val += self.dotproduct_AB.get_stats() * (1. - 1./self.iteration)
            return val/self.var_A.get_stats()[0]
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
        self.iteration += 1
        self._increment_variance(data)
        self._increment(data)

        if self.second_order :
            nb_required_sim = (2+ 2*self.nb_parms)*self.nb_sim
            if len(data) < nb_required_sim:
                raise Exception(f'The sample size must have {nb_required_sim} rows to compute the second order term.')
            else :
                self._increment_secondorder(data)

    def _increment_secondorder(self,data) :
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:(2+self.nb_parms)*self.nb_sim]
        sample_C = data[(2+self.nb_parms)*self.nb_sim:]

        for d in data :
            self.mean_tot.increment(d)

        for j in range(self.nb_parms) :
            for i in range(self.nb_parms):
                self.dotproduct_EC[j][i].increment(sample_E[j], sample_C[i], shift = self.mean_tot.get_stats())
                
            self.iterative_shifted_mean_E[j].increment(sample_E[j], shift = self.mean_tot.get_stats())
        
        self.dotproduct_AB.increment(sample_A, sample_B, shift = self.mean_tot.get_stats())
        self.iterative_shifted_mean_A.increment(sample_A, shift = self.mean_tot.get_stats())
        self.iterative_shifted_mean_B.increment(sample_B, shift = self.mean_tot.get_stats())

        val = np.array([[self.dotproduct_EC[j][i].get_stats() for i in range(self.nb_parms)] for j in range(self.nb_parms)])
        logger.info(f'val; {val}')
    def _increment(self, data : np.array) -> None :
        """
            Function that applies the specific iterative formula to increment the first and total order indices.
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

