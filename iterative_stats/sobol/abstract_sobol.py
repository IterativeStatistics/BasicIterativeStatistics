import numpy as np
from typing import Dict

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger



class IterativeAbstractSobol(AbstractIterativeStatistics):
    """
        Abstract class to compute Sobol indices
    """

    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.nb_parms = conf.get('nb_parms')
        self.nb_sim = conf.get('nb_sim', 1)
        self.var_A = IterativeVariance(conf)
 
    def _increment_variance(self, data):        
        self.var_A.increment(data[:self.nb_sim])
        

    def getIteration(self):
        return self.iteration

    def getFirstOrderSobol(self) :
        return self._compute_varianceI()/self.var_A.get_stats()[0]
 
    def getTotalOrderSobol(self) :
        return self._compute_VTi()/self.var_A.get_stats()[0]

    def increment(self, data) :
        raise Exception('Not implemented method')

    def _compute_varianceI(self):
        raise Exception('Not implemented method')

    def _compute_VTi(self):
        raise Exception('Not implemented method')

