from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class AbstractExperiment(ABC):
    def __init__(self, conf) -> None:
        self.seed = conf.get('seed', )
        self.sample_size = conf.get('sample_size')
        self.name = conf.get('name')
        pass

    def get_name(self):
        return self.name 

    def generator(self) :
        for _ in range(self.sample_size):
            yield self.draw()

    @abstractmethod
    def draw(self) -> np.array:
        pass