import numpy as np 
from iterative_stats.utils.logger import logger

from tests.mock.sensitivity.abstract_check import CheckSensitivityIndices

class SaltelliCheckSensitivityIndices(CheckSensitivityIndices):

    def compute_firstorderindices(self):
        if self.iteration > 1 :
            sample = np.append(self.data_A, self.data_B)
            for p in range(self.nb_parms):              
                sample = np.append(sample, self.data_E[:,p])

            mean = np.mean(sample)
            vi = [np.dot(self.data_B - mean, self.data_E[:,p] - mean)/(self.iteration - 1) - np.mean(self.data_A - mean) * np.mean(self.data_B - mean) for p in range(self.nb_parms)]
            var = np.var(self.data_A - mean, ddof = 1)
            logger.info(f'vi (gt) : {vi}')
            logger.info(f'var (gt): {var}')
            return vi/var
        else :
            return None 

    def compute_totalorderindices(self):
        if self.iteration > 1 :
            sample = np.append(self.data_A, self.data_B)
            for p in range(self.nb_parms):
                sample = np.append(sample, self.data_E[:,p])
            mean = np.mean(sample)
            var = np.var(self.data_A, ddof = 1)
            mean_prod = np.mean(self.data_A - mean) * np.mean(self.data_A - mean)
            vti = [mean_prod + var - np.dot(self.data_A - mean, self.data_E[:,p] - mean)/(self.iteration - 1)  for p in range(self.nb_parms)]    
            return  vti/var
        else :
            return None 


