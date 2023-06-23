import numpy as np
import numpy.typing as npt
from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics


class IterativeMoments(AbstractIterativeStatistics):
    """
    Iterative higher order moments following https://arxiv.org/pdf/1510.04923.pdf
    "Simpler Online Updates for Arbitrary-Order Central Moments"
    Takes:
    :param max_order: the maximum order of the moments to compute
    """
    def __init__(self, max_order: int, **kwargs):
        super().__init__(**kwargs)
        self.max_order: int = max_order

        # Cascading moments depending on the requested max_order
        self.m1: npt.NDArray[np.float_] = np.zeros(self.dimension)
        if self.max_order > 1:
            self.m2: npt.NDArray[np.float_] = np.zeros(self.dimension)
            self.theta2 = np.zeros(np.size(self.dimension))
        if self.max_order > 2:
            self.m3: npt.NDArray[np.float_] = np.zeros(self.dimension)
            self.theta3 = np.zeros(np.size(self.dimension))
        if self.max_order > 3:
            self.m4: npt.NDArray[np.float_] = np.zeros(self.dimension)
            self.theta4 = np.zeros(self.dimension)
        if self.max_order > 4:
            raise NotImplementedError("Moments of order > 4 not implemented")

    def increment(self, np_data: npt.NDArray[np.float_]):

        self.iteration += 1
        delta = self.get_delta(self.m1, np_data)
        delta_n = delta / self.iteration
        self.m1 += delta_n

        if self.max_order > 1:
            self.m2 += delta * ( delta - delta_n )
        if self.max_order > 2:
            delta_2 = delta * delta
            delta_n_2 = delta_n * delta_n
            self.m3 += -3.0 * delta_n * self.m2 + delta * ( delta_2 - delta_n_2 )
        if self.max_order > 3:
            self.m4 += -4.0 * delta_n * self.m3 - 6.0 * delta_n_2 * self.m2 \
                + delta * ( delta * delta_2 - delta_n * delta_n_2 )

    def get_delta(self, moment: npt.NDArray[np.float_], np_data):
        return np_data - moment

    def get_mean(self):
        return self.m1

    def get_variance(self):
        return self.m2 / self.iteration

    def get_skewness(self):
        return self.iteration ** 0.5 * self.m3 / self.m2 ** 1.5

    def get_kurtosis(self):
        return self.iteration * self.m4 / self.m2 ** 2

    def save_state(self):
        """
        This function saves the state of the moments
        """
        return {"m1": self.m1,
                "m2": self.m2,
                "m3": self.m3,
                "m4": self.m4,
                "increment": self.iteration}

    def load_from_state(self, state: dict):

        self.m1 = state["m1"]
        self.m2 = state["m2"]
        self.m3 = state["m3"]
        self.m4 = state["m4"]
        self.iteration = state["increment"]
