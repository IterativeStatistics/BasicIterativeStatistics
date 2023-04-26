
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractIterativeStatistics(ABC):
    """
        A basic abstract class to compute iteratives statistics.
    """
    def __init__(self, dim:int):
        self.state = np.zeros(dim)
        self.iteration = 0
        self.dimension = dim

    @abstractmethod
    def increment(self, data : np.array):
        """
            An abstract method to implement. It must contain the algorithm to compute iteratively statistics.
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