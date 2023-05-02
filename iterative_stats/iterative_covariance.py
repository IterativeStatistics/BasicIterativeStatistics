from typing import Dict 
import numpy as np
import copy

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.utils.logger import logger 

class IterativeCovariance(AbstractIterativeStatistics):
    def __init__(self, dim : int = 1, state: object = None):
        self.mean_1 = IterativeMean(dim)
        self.mean_2 = IterativeMean(dim)
        super().__init__(dim, state)

    def increment(self, data_1, data_2):
        # update mean
        self.iteration += 1
        prev_mean_1 = copy.deepcopy(self.mean_1.get_stats())
        prev_mean_2 = copy.deepcopy(self.mean_2.get_stats())
        self.mean_1.increment(data_1)
        self.mean_2.increment(data_2)
        
        # update covariance
        if self.iteration > 1:
            self.state = self.state * (self.iteration - 2)
            x = np.multiply(data_1 - prev_mean_1, data_2 - prev_mean_2)
            diff_mean = np.multiply(prev_mean_1 - self.mean_1.get_stats(), prev_mean_2 - self.mean_2.get_stats()) 
            self.state = self.state + x - diff_mean * self.iteration
            self.state = self.state / (self.iteration - 1)

    def getCovariance(self):
        return self.state

    def get_mean1(self):
        return self.mean_1.get_stats()
    
    def get_mean2(self):
        return self.mean_2.get_stats()

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['mean_1'] = self.mean_1.save_state()
        state['mean_2'] = self.mean_2.save_state()
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.mean_1 = IterativeMean(self.dimension, state=state.get('mean_1', None))
        self.mean_2 = IterativeMean(self.dimension, state=state.get('mean_2', None))