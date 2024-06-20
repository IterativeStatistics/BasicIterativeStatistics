from abc import ABC, abstractmethod
from typing import Deque
from collections import deque

import numpy as np
import numpy.typing as npt
from scipy.stats import qmc
import copy
import random


class AbstractExperiment(ABC):
    def __init__(self, nb_parms: int, nb_sim : int, apply_pick_freeze: bool = False,
                 seed : int = 0, second_order: bool = False, **kwargs) -> None:
        self.seed = seed
        self.nb_sim = nb_sim
        self.nb_parms = nb_parms
        self.apply_pick_freeze = apply_pick_freeze
        self.second_order = second_order

    def generator(self) :
        for _ in range(self.nb_sim):
            sample_A = self.draw()
            if self.apply_pick_freeze:
                sample_B = self.draw()
                sample = self.pick_freeze(sample_A, sample_B)
            else:
                sample = sample_A
            yield sample

    def pick_freeze(self, sample_A: npt.NDArray, sample_B : npt.NDArray) -> npt.NDArray:
        """
            Apply the pick-freeze method and construct the set of inputs parameters
        """
        if sample_A is None :
            sample_A = self.draw()
        if sample_B is None :
            sample_B = self.draw()
        sample = np.vstack(([sample_A], [sample_B]))
        for k in range(self.nb_parms):
            sample_Ek = copy.deepcopy(sample_A)
            sample_Ek[k] = sample_B[k]
            sample = np.vstack((sample, sample_Ek))

        if self.second_order :
            for k in range(self.nb_parms):
                sample_Ck = copy.deepcopy(sample_B)
                sample_Ck[k] = sample_A[k]
                sample = np.vstack((sample, sample_Ck))
        return sample

    @abstractmethod
    def draw(self) -> npt.NDArray[np.float_]:
        pass


class RandomUniform(AbstractExperiment):
    """
        Draw a uniform random number for a each parameter within the defined range
    """

    def __init__(self, l_bounds: list = [], u_bounds: list = [], **kwargs):
        super().__init__(**kwargs)
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        if len(l_bounds) == 1:
            self.l_bounds = [l_bounds[0] for _ in range(self.nb_parms)]
            self.u_bounds = [u_bounds[0] for _ in range(self.nb_parms)]

    def draw(self):
        param_set = []
        for n, _ in enumerate(range(self.nb_parms)):
            param_set.append(random.uniform(self.l_bounds[n], self.u_bounds[n]))
        # return param_set
        return np.array(param_set)


class Uniform3D(AbstractExperiment):
    """
        Draw a 3D uniform law based on the Latin Hypercube Sampling method (LHS).
        Use the qmc method from scipy
    """

    def __init__(self, l_bounds: list = [], u_bounds: list = [], **kwargs):
        super().__init__(**kwargs)
        self.sampler = qmc.LatinHypercube(d=self.nb_parms)
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        if len(l_bounds) == 1:
            self.l_bounds = [l_bounds[0] for _ in range(self.nb_parms)]
            self.u_bounds = [u_bounds[0] for _ in range(self.nb_parms)]

    def draw(self):
        sample = self.sampler.random(n=1)
        if not self.l_bounds or not self.u_bounds:
            return sample[0]
        else :
            return qmc.scale(sample, self.l_bounds, self.u_bounds)[0]


class HaltonGenerator(AbstractExperiment):
    """
    Deterministic sample generator based on scipy Halton sequence
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html
    """

    def __init__(self, l_bounds: list = [], u_bounds: list = [], **kwargs):
        super().__init__(**kwargs)
        self.parameters: Deque = deque()
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        if len(l_bounds) == 1:
            self.l_bounds = [l_bounds[0] for _ in range(self.nb_parms)]
            self.u_bounds = [u_bounds[0] for _ in range(self.nb_parms)]
        elif not l_bounds or not u_bounds:
            raise ValueError("Bounds must be defined for each parameter")

        self.sampler = qmc.Halton(d=self.nb_parms, scramble=False)
        self.populate_parameters()

    def populate_parameters(self):
        for _ in range(self.nb_sim):
            parameters = qmc.scale(self.sampler.random(1), self.l_bounds, self.u_bounds).tolist()
            self.parameters.extend(parameters)

    def add_parameters(self):
        parameters = qmc.scale(self.sampler.random(1), self.l_bounds, self.u_bounds).tolist()
        self.parameters.extend(parameters)

    def draw(self):
        try:
            return np.array(self.parameters.popleft())
        except IndexError:
            # this situation can arise if multiple failures require to draw a new parameter
            # for a given client or if additional clients are launched later in the study
            self.add_parameters()
            return np.array(self.parameters.popleft())


class LHSGenerator(AbstractExperiment):
    """
    Non-deterministic sample generator based on scipy LHS sampling
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html
    """

    def __init__(self, l_bounds: list = [], u_bounds: list = [], **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed=self.seed)
        self.parameters: Deque = deque()
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

        self.sampler = qmc.LatinHypercube(d=self.nb_parms, scramble=True, seed=self.rng)
        self.populate_parameters()

    def populate_parameters(self):
        for _ in range(self.nb_sim):
            parameters = qmc.scale(self.sampler.random(1), self.l_bounds, self.u_bounds).tolist()
            self.parameters.extend(parameters)

    def add_parameters(self):
        parameters = qmc.scale(self.sampler.random(1), self.l_bounds, self.u_bounds).tolist()
        self.parameters.extend(parameters)

    def draw(self):
        try:
            return np.array(self.parameters.popleft())
        except IndexError:
            # this situation can arise if multiple failures require to draw a new parameter
            # for a given client or if additional clients are launched later in the study
            self.add_parameters()
            return np.array(self.parameters.popleft())