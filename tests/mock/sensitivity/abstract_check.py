import numpy as np 
from abc import ABC, abstractmethod

from iterative_stats.utils.logger import logger

class CheckSensitivityIndices(ABC):
    def __init__(self, nb_parms, nb_sim = 1, second_order: bool = False):
        self.data_A = np.array([])
        self.data_B = np.array([])
        self.data_E = None
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.iteration = 0
        self.second_order = second_order
        if second_order :
            self.data_C = None

    def collect(self, sample) :
        self.iteration += 1
        self.data_A = np.append(self.data_A, sample[:self.nb_sim])
        self.data_B = np.append(self.data_B, sample[self.nb_sim:2*self.nb_sim])
        if self.data_E is None :
            self.data_E = np.array([sample[2*self.nb_sim:(2 + self.nb_parms)*self.nb_sim]])
        else :
            self.data_E = np.vstack((self.data_E, sample[2*self.nb_sim:(2 + self.nb_parms)*self.nb_sim]))
        if self.second_order :
            if self.data_C is None : 
                self.data_C = np.array([sample[(2 + self.nb_parms)*self.nb_sim:]])
            else: 
                self.data_C = np.vstack((self.data_C, sample[(2 + self.nb_parms)*self.nb_sim:]))
        
    def _compute_dotproduct(self, data_1, data_2):
        vec = []
        for p in range(self.nb_parms):
            vec.append(np.dot(data_2[:,p]-data_1, data_2[:,p]-data_1))
        return np.array(vec)
        
    def compute_secondorderindices(self):
        if not self.second_order :
            return None 
            
        # Compute global mean
        sample = np.append(self.data_A, self.data_B)
        for p in range(self.nb_parms):  
            sample = np.append(sample, self.data_E[:,p])
            sample = np.append(sample, self.data_C[:,p])

        mean_tot = np.mean(sample)
        dotprod_AB = np.dot(self.data_A - mean_tot , self.data_B - mean_tot)/self.iteration 
        
        var = np.var(self.data_A - mean_tot, ddof = 1)
        final = np.zeros((self.nb_parms, self.nb_parms))
        first_order = self.compute_firstorderindices()
        if first_order is None :
            return None 
        
        for i in range(self.nb_parms):
            for j in range(self.nb_parms) : 
                final[i][j] = np.dot(self.data_E[:,i] - mean_tot, self.data_C[:,j] - mean_tot)/(self.iteration - 1)
                final[i][j] += dotprod_AB
                final[i][j] /= var
                final[i][j] -= first_order[i] + first_order[j]
        logger.info(f'here (gt): {final}')
        return final
    
    @abstractmethod
    def compute_firstorderindices(self):
        raise Exception('Not implemented method')

    @abstractmethod
    def compute_totalorderindices(self):
        raise Exception('Not implemented method')

