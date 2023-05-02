import numpy as np
import copy 

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeMean(AbstractIterativeStatistics):
    """
        Iterative Mean
    """

    def increment(self, data):
        self.iteration += 1
        self.state += (data - self.state) / float(self.iteration)
        logger.debug(f'increment= {self.increment}, mean= {self.state}')


class IterativeShiftedMean(AbstractIterativeStatistics):
    """
        Iterative Mean with a shift in the data
    """
    def __init__(self, dim : int = 1, state: object = None):
        self.previous_shift = None
        super().__init__(dim, state)


    def increment(self, data, shift: np.array):
        self.iteration += 1
        if self.iteration == 1 : # Initialization
            self.state += data - shift 
        else :
            # logger.info(f'------------ data: {data}')
            self.state *= (1. - 1./self.iteration)
            self.state += (data - self.previous_shift)/self.iteration  + (self.previous_shift - shift)

        # update the shift
        self.previous_shift = copy.deepcopy(shift)

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['previous_shift'] = self.previous_shift
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.previous_shift = state.get('previous_shift', None)