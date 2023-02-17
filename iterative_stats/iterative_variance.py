from typing import Dict 
import numpy as np

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.utils.logger import logger 

class IterativeVariance(AbstractIterativeStatistics):
    def __init__(self, vector_size:int = 1):
        super().__init__(vector_size)
        self.mean = IterativeMean(vector_size)
        self.sumOfCenteredSquares = np.zeros(vector_size)

    def increment(self, data):
        self.iteration += 1

        if self.iteration > 1 : 
            self.sumOfCenteredSquares += (
                (self.iteration - 1) * (data - self.mean.get_stats()) ** 2 / self.iteration
            )

        # update mean 
        self.mean.increment(data)

        # compute variance
        if self.iteration > 1 :
            self.state = self.sumOfCenteredSquares / (self.iteration - 1)
        logger.debug(f'increment= {self.increment}, variance= {self.state}')

    def get_variance(self):
        return self.state 

    def get_mean(self):
        return self.mean.get_stats()