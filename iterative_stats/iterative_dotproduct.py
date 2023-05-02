from typing import Dict 
import numpy as np
import copy

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeShiftedMean
from iterative_stats.utils.logger import logger 
from iterative_stats.utils.dot_prod import multi_dim_dotproduct

class IterativeDotProduct(AbstractIterativeStatistics):
    def __init__(self, dim : int = 1, iterative_shifted_mean_1 : IterativeShiftedMean = None, 
                        iterative_shifted_mean_2 : IterativeShiftedMean = None, state: object = None):
        """
            Compute iteratively the formula sum_k=0:N (A_k - shift_N)(B_k - shift_N)/(N-1)
            iterative_shifted_mean_1 (IterativeShiftedMean) : add an external iterative shifted mean. If it is not None, the mean will not be updated into this class
        """
        self.previous_shift = None

        if iterative_shifted_mean_1 is None :
            self.iterative_shifted_mean_1 = IterativeShiftedMean(dim)
            self.external_mean_1 = False
        else :
            self.iterative_shifted_mean_1 = iterative_shifted_mean_1
            self.external_mean_1 = True

        if iterative_shifted_mean_2 is None :
            self.iterative_shifted_mean_2 = IterativeShiftedMean(dim)
            self.external_mean_2 = False
        else :
            self.iterative_shifted_mean_2 = iterative_shifted_mean_2
            self.external_mean_2 = True

        super().__init__(dim, state)
        self.data_1 = None
        self.data_2 = None

    def increment(self, data_1, data_2, shift):
        self.iteration += 1
        
        if self.iteration == 1 :
            self.data_1 = data_1
            self.data_2= data_2
        elif self.iteration == 2 :
            if self.dimension == 1 :
                self.data_1 = np.array([self.data_1, data_1])
                self.data_2= np.array([self.data_2, data_2])
            else :
                self.data_1 = np.vstack((self.data_1, data_1))
                self.data_2 = np.vstack((self.data_2, data_2))
            self.state = multi_dim_dotproduct(self.data_1 - shift, self.data_2 - shift, self.dimension)
            del self.data_1 , self.data_2
        else :
            diff_shift = self.previous_shift - shift
            self.state *= (1 - 1/(self.iteration - 1))
            self.state += np.multiply(data_1 - self.previous_shift, data_2 - self.previous_shift)/(self.iteration - 1)
            val = self.iterative_shifted_mean_1.get_stats() + self.iterative_shifted_mean_2.get_stats()
            val +=  diff_shift + (data_1 + data_2 - shift - self.previous_shift)/(self.iteration - 1)
            self.state += val * diff_shift
            
            
        # Update iterative shifted mean if not external
        if not self.external_mean_1 :
            self.iterative_shifted_mean_1.increment(data_1, shift)

        if not self.external_mean_2 :
            self.iterative_shifted_mean_2.increment(data_2, shift)
        
        self.previous_shift = copy.deepcopy(shift)


    def get_mean_1(self):
        return self.iterative_shifted_mean_1.get_stats()

    def get_mean_2(self):
        return self.iterative_shifted_mean_2.get_stats()

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['previous_shift'] = self.previous_shift
        if not self.external_mean_1 :
            state['external_mean_1'] = self.iterative_shifted_mean_1.save_state()
        if not self.external_mean_2 :
            state['external_mean_2'] = self.iterative_shifted_mean_2.save_state()
        return state
        

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.previous_shift = state.get('previous_shift')
        if state.get('external_mean_1', None) is not None :
            self.external_mean_1 = False
            self.iterative_shifted_mean_1 = IterativeShiftedMean(self.dimension, state.get('external_mean_1'))
        if state.get('external_mean_2', None) is not None :
            self.external_mean_2 = False
            self.iterative_shifted_mean_2 = IterativeShiftedMean(self.dimension, state.get('external_mean_2'))
