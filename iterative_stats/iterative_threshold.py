import numpy as np
import copy 

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeThreshold(AbstractIterativeStatistics):
    """
        Iterative Threshold
    """

    def __init__(self, dim:int = 1, min_threshold: np.array = None, max_threshold: np.array = None, state: object = None):
        super().__init__(dim, state)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def increment(self, data):
        self.iteration += 1

        if self.min_threshold is not None:
            res_min = data >= self.min_threshold
        else :
            res_min = np.ones(self.dim)
        if self.max_threshold is not None:
            res_max = data <= self.max_threshold
        else :
            res_max = np.ones(self.dim)

        self.state += res_min & res_max
        
