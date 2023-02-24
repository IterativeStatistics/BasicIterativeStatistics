import numpy as np
import copy 

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeMean(AbstractIterativeStatistics):

    def increment(self, data):
        self.iteration += 1
        self.state += (data - self.state) / float(self.iteration)
        logger.debug(f'increment= {self.increment}, mean= {self.state}')

class IterativeShiftedMean(AbstractIterativeStatistics):
    def __init__(self, vector_size : int = 1):
        super().__init__(vector_size)
        self.previous_shift = None

    def increment(self, data, shift: np.array):
        self.iteration += 1
        if self.iteration == 1 : # Initialization
            self.state = data - shift 
        else :
            # logger.info(f'------------ data: {data}')
            self.state *= (1. - 1./self.iteration)
            self.state += (data - self.previous_shift)/self.iteration  + (self.previous_shift - shift)

        # update the shift
        self.previous_shift = copy.deepcopy(shift)
