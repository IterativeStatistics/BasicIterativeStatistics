# Iterative quantile estimation by use of a Robbins-Monro algorithm
# Implementer: Alejandro Ribés

import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeQuantile(AbstractIterativeStatistics):
    def __init__(self, dim: int = 1, state: object = None, alpha: float = 0.5, max_it: int = 0):
        if state is None :
            self.state_min = np.zeros(dim)
            self.state_max = np.zeros(dim)
            self.indicator = np.zeros(dim)
            self.C = np.zeros(dim)
        super().__init__(dim, state)

        self.max_it = max_it
        self.alpha = alpha   


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
        

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['indicator'] = self.indicator
        state['C'] = self.C
        state['state_min'] = self.state_min
        state['state_max'] = self.state_max
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.state_min = state.get('state_min')
        self.state_max = state.get('state_max')
        self.indicator = state.get('indicator')
        self.C = state.get('C')