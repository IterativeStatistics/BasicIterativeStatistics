import numpy as np 
from abc import ABC, abstractmethod

class CheckSensitivityIndices(ABC):
    def __init__(self, nb_parms, nb_sim = 1):
        self.data_A = np.array([])
        self.data_B = np.array([])
        self.data_E = np.array([]*nb_parms)
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.iteration = 0

    def collect(self, sample) :
        self.iteration += 1

        if self.data_B.size == 0 and self.data_E.size == 0 :
            self.data_A = sample[:self.nb_sim]
            self.data_B = sample[self.nb_sim:2*self.nb_sim]
            self.data_E = sample[2*self.nb_sim:]
        else :
            self.data_A = np.append(self.data_A, sample[:self.nb_sim])
            self.data_B = np.append(self.data_B, sample[self.nb_sim:2*self.nb_sim])
            self.data_E = np.vstack((self.data_E, sample[2*self.nb_sim:]))

    def _compute_dotproduct(self, data_1, data_2):
        vec = []
        for p in range(self.nb_parms):
            vec.append(np.dot(data_2[:,p]-data_1, data_2[:,p]-data_1))
        return np.array(vec)
        
    @abstractmethod
    def compute_firstorderindices(self):
        raise Exception('Not implemented method')

    @abstractmethod
    def compute_totalorderindices(self):
        raise Exception('Not implemented method')

