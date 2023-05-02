import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeExtrema(AbstractIterativeStatistics):
    def __init__(self, dim: int = 1, state: object = None):
        self.state_min = np.zeros(dim)
        super().__init__(dim, state)
        

    def increment(self, data):
        self.state_min = np.minimum(data, self.state_min)
        self.state = np.maximum(data, self.state)

    def get_min(self):
        return self.state_min

    def get_max(self):
        return self.state 

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['state_min'] = self.state_min
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.state_min = state.get('state_min', None)