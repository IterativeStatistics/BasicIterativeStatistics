import numpy as np
from typing import Dict

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger



class IterativeSensitivityJansen(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices based on the Jansen estimate
    """
    def __init__(self, nb_parms: int, vector_size: int = 1):
        super().__init__(nb_parms = nb_parms, nb_sim = 1, vector_size = vector_size)
        self.mean_tot = IterativeMean(vector_size)
        self.state = {'sumAminusE' : np.zeros(self.nb_parms), 'sumBminusE' : np.zeros(self.nb_parms)}
       
    def increment(self, data):
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]
       
        self.iteration += 1

        self._increment_variance(data)

        for p in range(self.nb_parms):
            # update last order
            self.state['sumAminusE'][p] += np.dot(sample_A- sample_E[p], sample_A - sample_E[p])
            self.state['sumBminusE'][p] += np.dot(sample_B- sample_E[p], sample_B - sample_E[p])
            

    def getSobol(self):
        return self.sobol

    def _compute_varianceI(self) :
        val = self.var_A.get_stats()[0] - self.state.get('sumBminusE')/(2*self.iteration - 1)
        return val


    def _compute_VTi(self) :
        coeff = 2*self.iteration - 1
        return self.state.get('sumAminusE')/coeff

    def getIteration(self):
        return self.iteration