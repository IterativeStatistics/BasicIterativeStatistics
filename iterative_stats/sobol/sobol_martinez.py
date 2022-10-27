from cmath import isnan

import numpy as np
from typing import Dict
import copy

from src.abstract_iterative_statistcs import AbstractIterativeStatistics
from src.iterative_variance import IterativeVariance
from src.iterative_covariance import IterativeCovariance
from src.utils.logger import logger


class IterativeSobol(AbstractIterativeStatistics):
    """
    Estimates the Sobol indices iteratively with a robust formula.
    """
    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.nb_parms = conf.get('nb_parms')
        self.varData_B = IterativeVariance(conf)
        self.varData_C = [IterativeVariance(conf) for _ in range(self.nb_parms)]
        self.covData_BC = [IterativeCovariance(conf) for _ in range(self.nb_parms)]
        self.state = np.zeros(self.nb_parms)
        
    def increment(self, data_1, data_2):
        # update mean
        self.iteration += 1
        self.varData_B.increment(data_1)
        for k in range(self.nb_parms):
            self.varData_C[k].increment(data_2[k])
            # update covariance
            self.covData_BC[k].increment(data_1,data_2[k])
            var_prod = np.multiply(self.varData_C[k].get_stats(), self.varData_B.get_stats())
            self.state[k] = np.divide(self.covData_BC[k].get_stats(), np.sqrt(var_prod))
            
    def getSobol(self):
        return self.sobol

    def getIteration(self):
        return self.iteration

