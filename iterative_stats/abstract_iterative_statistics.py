
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractIterativeStatistics(ABC):
    """
        A basic abstract class to compute iteratives statistics.
    """
    def __init__(self, dim:int, state: object = None):
        self.dimension = dim
        if state is None :
            self.state = np.zeros(dim)
            self.iteration = 0
        else :
            self.load_from_state(state)
        

    @abstractmethod
    def increment(self, data : np.array):
        """
            An abstract method to implement. It must contain the algorithm to compute iteratively statistics.
        """
        raise Exception('Not implemented method')
    
    def save_state(self):
        """
            A method that save the current state.
        """
        return {'iteration': self.iteration, 'state': self.state}

    def load_from_state(self, state: object):
        """
            A method that load the current state
        """
        self.iteration = state.get('iteration', 0)
        self.state = state.get('state', None)

    def get_stats(self):
        """
            A method that return the current state
        """
        return self.state
    
    def get_iteration(self):
        """
            A method that return the current iteration value
        """
        return self.iteration