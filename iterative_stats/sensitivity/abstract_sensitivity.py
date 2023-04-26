import numpy as np
from typing import Dict

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeShiftedMean, IterativeMean
from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger
from iterative_stats.sensitivity import SALTELLI


class IterativeAbstractSensitivity(AbstractIterativeStatistics):
    """
        Abstract class to compute the first, second and total sensitivity indices
    """

    def __init__(self, nb_parms : int, dim: int = 1, 
                        second_order: bool = False, name: str = "") -> None:
        """
            nb_parms (int) : number of input variables
            dim (int) : output size
            second_order (bool) : a boolean indicating if the second order must be computed or not. 
        """
        super().__init__(dim)
        self.nb_parms : int = nb_parms
        self.second_order : bool = second_order
        self.var_A: AbstractIterativeStatistics = IterativeVariance(dim)
        self.name : str = name 
        
        self.mean_tot : IterativeMean = IterativeMean(dim)
        self.mean_tot_iteration : int = 0

        if self.second_order or self.name == SALTELLI:
            self.iterative_shifted_mean_A = IterativeShiftedMean(dim)
            self.iterative_shifted_mean_B = IterativeShiftedMean(dim)
            self.iterative_shifted_mean_E = [IterativeShiftedMean(dim) for _ in range(self.nb_parms)]

        if self.second_order :
            self.dotproduct_AB = IterativeDotProduct(dim, iterative_shifted_mean_1 = self.iterative_shifted_mean_A, iterative_shifted_mean_2 = self.iterative_shifted_mean_B)
            self.dotproduct_EC = [[IterativeDotProduct(dim, iterative_shifted_mean_1 = self.iterative_shifted_mean_E[i]) for _ in range(self.nb_parms)] for i in range(self.nb_parms)]

    def _increment_variance(self, data : np.array) -> None: 
        """
            data (np.array) : input data
            Function that increments the variance of sample A.
        """       
        self.var_A.increment(data[0])
        
    def getIteration(self) -> int:
        """
            Get the current iteration index.
        """
        return self.iteration

    def getFirstOrderIndices(self) -> np.array:
        """
            Compute the self.nb_parms first order sensitivity indices
        """
        if self.iteration > 1 :
            return self._compute_varianceI()/self.var_A.get_stats()[:,None]
        else :
            return None
 
    def getSecondOrderIndices(self) -> np.array:
        """
            Compute the self.nb_parms second order sensitivity indices
        """
        if self.iteration > 1 and self.second_order :   
            first_order = self._compute_varianceI()
            val = np.zeros((self.dimension, self.nb_parms, self.nb_parms))
            
            for i in range(self.nb_parms):
                for j in range(self.nb_parms):
                    val[:, i,j] = self.dotproduct_EC[i][j].get_stats() - first_order[:,i] - first_order[:,j] 
            val += self.dotproduct_AB.get_stats()[:, None, None] * (1. - 1./self.iteration)
            return val/self.var_A.get_stats()[:, None, None]
        else :
            return None

    def getTotalOrderIndices(self) -> np.array :
        """
            Compute the self.nb_parms total order sensitivity indices
        """
        if self.iteration > 1 :
            return np.divide(self._compute_VTi(),self.var_A.get_stats()[:,None])
        else :
            return None

    def increment(self, data : np.array) -> None :
        """
            Function that applies the iterative formula to increment the first and total order indices.
        """
        self.iteration += 1
        # Update the total mean
        if self.second_order or self.name == SALTELLI:
            self._update_global_mean(data)

        # Update the variance (sample A)
        self._increment_variance(data)

        # Update the specific increment data
        self._increment(data)


        if self.second_order :
            nb_required_sim = (2+ 2*self.nb_parms)
            if len(data) < nb_required_sim:
                raise Exception(f'The sample size must have {nb_required_sim} rows to compute the second order term.')
            else :
                self._increment_dotproduct(data)
        
        # Update all the shifted mean 
        if self.second_order or self.name == SALTELLI :
            self.iterative_shifted_mean_A.increment(data[0], shift = self.mean_tot.get_stats())
            self.iterative_shifted_mean_B.increment(data[1], shift = self.mean_tot.get_stats())
            for i in range(self.nb_parms):
                self.iterative_shifted_mean_E[i].increment(data[2:(2+self.nb_parms)][i], shift = self.mean_tot.get_stats())


    def _increment(self, data : np.array) -> None :
        """
            Function that applies the specific iterative formula to increment the first and total order indices.
        """
        raise Exception('Not implemented method')


    def _increment_dotproduct(self,data : np.array) -> None :
        """
            data (np.array) : a (self.nb_sim size, self.nb_parms) np.array vector
            Internal method that increments the dot products order sensitivity value.
        """

        # Update EC dot product and E iterative mean
        sample_E = data[2:2+self.nb_parms]
        sample_C = data[2+self.nb_parms:]
        for i in range(self.nb_parms) :
            for j in range(self.nb_parms):
                self.dotproduct_EC[i][j].increment(sample_E[i], sample_C[j], shift = self.mean_tot.get_stats())    
        
        # Update AB dot product
        self.dotproduct_AB.increment(data[0], data[1], shift = self.mean_tot.get_stats())
    
    def _update_global_mean(self, data : np.array) -> None :
        if self.iteration -1 == self.mean_tot_iteration : 
            # Update the global mean of the system
            for d in data :
                self.mean_tot.increment(d)
            self.mean_tot_iteration += 1 

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

