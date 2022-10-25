from src.abstract_iterative_statistcs import AbstractIterativeStatistics
from src.utils.logger import logger 

class IterativeMean(AbstractIterativeStatistics):

    def increment(self, data):
        self.iteration += 1
        self.state += (data - self.state) / float(self.iteration)
        logger.debug(f'increment= {self.increment}, mean= {self.state}')