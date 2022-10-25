
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractIterativeStatistics(ABC):
    def __init__(self, conf: Dict):
        vector_size = conf.get('vector_size')
        self.state = np.zeros(vector_size)
        self.iteration = 0
        self.dimension = vector_size

    @abstractmethod
    def increment(self, data : np.array):
        raise Exception('Not implemented method')
    
    def get_stats(self):
        return self.state
    
    def get_iteration(self):
        return self.iteration