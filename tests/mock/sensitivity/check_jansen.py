import numpy as np 

# from iterative_stats.utils.logger import logger

from tests.mock.sensitivity.abstract_check import CheckSensitivityIndices

class JansenCheckSensitivityIndices(CheckSensitivityIndices):


    def _compute_centeredsquare(self, data):
        vec = []
        for _ in range(self.nb_parms):
            vec.append(np.dot(data, data))
        return np.array(vec)

    def getFirstOrderIndices(self):
        if self.iteration > 1 :
            var = np.var(self.data_A, ddof = 1)
            vi_first_order = var - self._compute_dotproduct(self.data_B, self.data_E)/(2*self.iteration - 1)
            return vi_first_order/var
        else :
            return None 

    def getTotalOrderIndices(self):
        if self.iteration > 1 :
            var = np.var(self.data_A, ddof = 1)
            vi_total_order = self._compute_dotproduct(self.data_A, self.data_E)/(2*self.iteration - 1)
            return vi_total_order/var
        else :
            return None 