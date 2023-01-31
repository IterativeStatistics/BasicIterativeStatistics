import numpy as np
from typing import Dict

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.iterative_covariance import IterativeCovariance
from iterative_stats.utils.logger import logger

def center_reduce(data):
    if np.std(data) > 0 :
        return (data - np.mean(data))/np.std(data)
    else :
        return data


class IterativeMartinezSobol(AbstractIterativeStatistics):
    """
    Estimates the Sobol indices iteratively with a robust formula.
    """
    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.nb_parms = conf.get('nb_parms')
        self.nb_sim = conf.get('nb_sim', 1)
        self.varData_A = IterativeVariance(conf)
        self.varData_B = IterativeVariance(conf)
        self.varData_E = [IterativeVariance(conf) for _ in range(self.nb_parms)]
       
        self.covData_AE = [IterativeCovariance(conf) for _ in range(self.nb_parms)]
        self.covData_BE = [IterativeCovariance(conf) for _ in range(self.nb_parms)]
       
        self.state = {'pearson_A' : np.zeros(self.nb_parms), 'pearson_B' : np.zeros(self.nb_parms)}
       
    def increment(self, data):
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]
        # reduce and center data_1 and data_2
        # data_1 = center_reduce(data_1)
        # data_2 = center_reduce(data_2)
        logger.info(f'sample: {sample_A}, B= {sample_B}, E= {sample_E}')
        # update mean
        self.iteration += 1
        self.varData_A.increment(sample_A)
        self.varData_B.increment(sample_B)
        for p in range(self.nb_parms):
            self.varData_E[p].increment(sample_E[p])

            # update first order
            self.covData_BE[p].increment(sample_B,sample_E[p])
            var_prod = np.multiply(self.varData_E[p].get_stats(), self.varData_B.get_stats())
            self.state['pearson_B'][p] = np.divide(self.covData_BE[p].get_stats(), np.sqrt(var_prod))

            # update last order
            self.covData_AE[p].increment(sample_A,sample_E[p])
            var_prod = np.multiply(self.varData_E[p].get_stats(), self.varData_A.get_stats())
            self.state['pearson_A'][p] = np.divide(self.covData_AE[p].get_stats(), np.sqrt(var_prod))
           
    def getSobol(self):
        return self.sobol

    def getFirstOrderSobol(self) :
        return self.state.get('pearson_B')

    def getTotalOrderSobol(self) :
        return 1 - self.state.get('pearson_A')

    def getIteration(self):
        return self.iteration