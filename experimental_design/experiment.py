from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import copy

from iterative_stats.utils.logger import logger 

class AbstractExperiment(ABC):
    def __init__(self, nb_parms: int, nb_sim : int, apply_pick_freeze: bool = True, seed : int = 0,  **kwargs) -> None:
        self.seed = seed
        self.nb_sim = nb_sim
        self.nb_parms = nb_parms
        self.apply_pick_freeze = apply_pick_freeze

    def generator(self) :
        for _ in range(self.nb_sim):
            sample_A = self.draw()
            sample_B = self.draw()
            pick_freeze_sample = self.pick_freeze(sample_A[0], sample_B[0])
            yield pick_freeze_sample
        

    def pick_freeze(self, sample_A: np.array, sample_B : np.array) -> np.array:
        """
            Apply the pick-freeze method and construct the set of inputs parameters
        """
        if sample_A is None :
            sample_A = self.draw()[0]
        if sample_B is None :
            sample_B = self.draw()[0]
        sample = np.vstack(([sample_A], [sample_B]))
        for k in range(self.nb_parms):
            sample_Ck = copy.deepcopy(sample_A)  
            sample_Ck[k] = sample_B[k]
            sample = np.vstack( (sample, sample_Ck))
        return sample 

    @abstractmethod
    def draw(self) -> np.array:
        pass

