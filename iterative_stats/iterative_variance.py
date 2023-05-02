from typing import Dict 
import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.utils.logger import logger 

class IterativeVariance(AbstractIterativeStatistics):
    def __init__(self, dim:int = 1, state: object = None):
        if state is None :
            self.mean = IterativeMean(dim)
            self.sumOfCenteredSquares = np.zeros(dim)
        super().__init__(dim, state)

    def increment(self, data):
        self.iteration += 1
        if self.iteration > 1 : 
            self.sumOfCenteredSquares += (
                (self.iteration - 1) * (data - self.mean.get_stats()) ** 2 / self.iteration
            )

        # update mean 
        self.mean.increment(data)

        # compute variance
        if self.iteration > 1 :
            self.state = self.sumOfCenteredSquares / (self.iteration - 1)
        logger.debug(f'increment= {self.increment}, variance= {self.state}')

    def get_variance(self):
        return self.state 

    def get_mean(self):
        return self.mean.get_stats()

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['mean'] = self.mean.save_state()
        state['sumOfCenteredSquares'] = self.sumOfCenteredSquares
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.mean = IterativeMean(self.dimension, state.get('mean'))
        self.sumOfCenteredSquares = state.get('sumOfCenteredSquares')