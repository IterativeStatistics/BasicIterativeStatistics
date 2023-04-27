# Iterative quantile estimation by use of a Robbins-Monro algorithm
# Implementer: Alejandro Ribés

import copy
import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeQuantile(AbstractIterativeStatistics):
    def __init__(self, dim: int = 1):
        super().__init__(dim)
        self.state_min = np.zeros(self.dimension)
        self.state_max = np.zeros(self.dimension)
        self.indicator = np.zeros(self.dimension)
        self.C = np.zeros(self.dimension)
        self.max_it = 0 
        self.alpha = 0.5    

    def setDesiredQuantile(self, q):
        self.alpha = q

    def setMaxIterations(self, mi):
        self.max_it = mi

    def increment(self, data):
        self.iteration += 1
        
        if self.iteration == 1:
            self.state_min = data
            self.state_max = data
        else:
            self.state_min = np.minimum(data, self.state_min)
            self.state_max = np.maximum(data, self.state_max)
            
        if self.max_it <= 1.0 :
            logger.error(f'MaxIterations= {self.max_it}, it has not been set (or set to <= 1)')

        gamma = 0.5 + 0.5 * ((self.iteration - 1.0)/(self.max_it - 1.0))
        C = abs(self.state_max - self.state_min)
        
        self.indicator = (data <= self.state).astype(np.int8)

        if self.iteration == 1:
            self.state = data
        else:
            self.state -= (C / self.iteration**gamma) * (self.indicator - self.alpha)
        
    def get_quantile(self):
        return self.state

