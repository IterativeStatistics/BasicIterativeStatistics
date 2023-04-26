import numpy as np

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.utils.logger import logger



class IterativeSensitivityJansen(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices based on the Jansen estimate
    """
    def __init__(self, nb_parms: int, dim: int = 1, second_order: bool = False):
        super().__init__(nb_parms = nb_parms, dim = dim, second_order=second_order)
        self.AminusE = np.zeros((dim, self.nb_parms))
        self.BminusE = np.zeros((dim, self.nb_parms))
        del self.state
       
    def _increment(self, data):
        sample_A = data[0]
        sample_B = data[1]
        sample_E = data[2:(2 + self.nb_parms)]
        for p in range(self.nb_parms):
            self.AminusE[:,p] += np.multiply(sample_A- sample_E[p], sample_A - sample_E[p])
            self.BminusE[:,p] += np.multiply(sample_B- sample_E[p], sample_B - sample_E[p])
        logger.info(f'Jansen: {self.AminusE}')

    def _compute_varianceI(self) :
        return self.var_A.get_stats()[:, None] - self.BminusE/(2*self.iteration - 1)

    def _compute_VTi(self) :
        coeff = 2*self.iteration - 1
        return self.AminusE/coeff

    def getIteration(self):
        return self.iteration