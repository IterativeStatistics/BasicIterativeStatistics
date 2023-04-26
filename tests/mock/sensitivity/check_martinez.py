import numpy as np 


from tests.mock.sensitivity.abstract_check import CheckSensitivityIndices

class MartinezCheckSensitivityIndices(CheckSensitivityIndices):

    def _compute_pearson(self, data_1, data_2):
        pearson = np.empty(self.nb_parms)
        for p in range(self.nb_parms) :
            cov = np.cov(data_1, data_2[:,p])[0][1]
            pearson[p] = cov / (np.std(data_1, ddof=1)*np.std(data_2[:,p], ddof=1))
        return pearson

    def getFirstOrderIndices(self):
        if self.iteration > 1 :
            return self._compute_pearson(self.data_B, self.data_E)
        else :
            return None 

    def getTotalOrderIndices(self):
        if self.iteration > 1 :
            return  1 - self._compute_pearson(self.data_A, self.data_E)
        else :
            return None 


