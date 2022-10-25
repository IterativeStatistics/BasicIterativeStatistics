from typing import Dict 
import copy
import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeExtrema(AbstractIterativeStatistics):
    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.state_min = copy.deepcopy(self.state)
        

    def increment(self, data):
        self.state_min = np.minimum(data, self.state_min)
        self.state = np.maximum(data, self.state)

    def get_min(self):
        return self.state_min

    def get_max(self):
        return self.state 