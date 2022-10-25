from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.utils.logger import logger 

class IterativeMean(AbstractIterativeStatistics):

    def increment(self, data):
        self.iteration += 1
        self.state += (data - self.state) / float(self.iteration)
        logger.debug(f'increment= {self.increment}, mean= {self.state}')