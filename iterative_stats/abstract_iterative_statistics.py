
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractIterativeStatistics(ABC):
    """
        A basic abstract class to compute iteratives statistics.
    """
    def __init__(self, dim:int, state: object = None):
        if state is None :
            self.state = np.zeros(dim)
            self.iteration = 0
        else :
            self.load_from_state(state)
        self.dimension = dim

    @abstractmethod
    def increment(self, data : np.array):
        """
            An abstract method to implement. It must contain the algorithm to compute iteratively statistics.
        """
        raise Exception('Not implemented method')
    
    @abstractmethod
    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        raise Exception('Not implemented method')

    @abstractmethod
    def load_from_state(self, state:object):
        """
            An abstract method to implement. It load the current state of the object.
        """
        raise Exception('Not implemented method')

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