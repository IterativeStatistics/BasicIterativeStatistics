
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractIterativeAlgorithm(ABC):
    def __init__(self, conf: Dict):
        self.dimension = conf.get('dimension')
        self.state = None
        self.iteration = 0

    @abstractmethod
    def update(self, data : np.array):
        raise Exception('Not implemented method')
    
    def get_stats(self):
        return self.state
    
    def get_iteration_number(self):
        return self.iteration