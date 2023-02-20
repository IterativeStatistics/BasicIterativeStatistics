from typing import Dict 
import numpy as np
import copy

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.utils.logger import logger 

class IterativeDotProduct(AbstractIterativeStatistics):
    def __init__(self, vector_size : int = 1):
        super().__init__(vector_size)

    def increment(self, data_1, data_2):
        self.iteration += 1
        self.state += data_1 * data_2
        