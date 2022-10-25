from typing import Dict 
import numpy as np

from src.abstract_iterative_statistcs import AbstractIterativeStatistics
from src.iterative_mean import IterativeMean
from src.utils.logger import logger 

class IterativeVariance(AbstractIterativeStatistics):
    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.mean = IterativeMean(conf)
        self.sumOfCenteredSquares = np.zeros(conf.get('vector_size'))

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