import numpy as np
from typing import Dict

from iterative_stats.sobol.abstract_sobol import IterativeAbstractSobol
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger



class IterativeJansenSobol(IterativeAbstractSobol):
    """
    Estimates the Sobol indices based on the Jansen estimate
    """
    def __init__(self, conf: Dict):
        super().__init__(conf)
        self.mean_tot = IterativeMean(conf)
        self.state = {'sumAminusE' : np.zeros(self.nb_parms), 'sumBminusE' : np.zeros(self.nb_parms),
                        'A_square': np.zeros(self.nb_parms)}
       
    def increment(self, data):
        sample_A = data[:self.nb_sim]
        sample_B = data[self.nb_sim:2*self.nb_sim]
        sample_E = data[2*self.nb_sim:]
        
        self.iteration += 1
        
        for d in data:
            self.mean_tot.increment(d)

        mean_A = self.var_A.get_mean()

        for p in range(self.nb_parms):
            # update last order
            self.state['sumAminusE'][p] += np.dot(sample_A- sample_E[p], sample_A - sample_E[p])
            self.state['sumBminusE'][p] += np.dot(sample_B- sample_E[p], sample_B - sample_E[p])
            self.state['A_square'][p] += np.dot(sample_A , sample_A)
            self.state['A_square'][p] += self.iteration * self.mean_tot.get_stats()**2
            self.state['A_square'][p] -= 2* self.mean_tot.get_stats() * mean_A
           
    def getSobol(self):
        return self.sobol

    def _compute_varianceI(self) :
        return self.state.get('A_square')/(self.iteration - 1) -self.state.get('sumBminusE')/(2*self.iteration - 1)

    def _compute_VTi(self) :
        coeff = 2*self.iteration - 1
        return self.state.get('sumAminusE')/coeff

    def getIteration(self):
        return self.iteration