import numpy as np
import numpy.typing as npt
from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics


class IterativeMoments(AbstractIterativeStatistics):
    """
    This class implements a data structure of moments similar to the one
    in the original Melissa version
    """
    def __init__(self, max_order: int, **kwargs):
        super().__init__(**kwargs)
        self.max_order: int = max_order
        if self.max_order <= 1:
            self.iterative_stat = IterativeMean(self.dimension)
        else:
            # take lowest primitive possible to avoid duplicating
            # the mean calculation
            self.iterative_stat = IterativeVariance(self.dimension)

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

    def increment(self, np_data: npt.NDArray[np.float_]):
        """
        This function increments the various orders of moments
        """
        self.iteration += 1

        self.iterative_stat.increment(np_data)
        if self.max_order <= 1:
            self.m1 = self.iterative_stat.get_stats()
        else:
            self.m1 = self.iterative_stat.get_mean()
            self.theta2 = self.iterative_stat.get_variance()
            self.m2 = self.theta2 + np.square(self.m1)

        if self.max_order > 2:
            self.m3 = self.increment_moments_mean(self.m3, np_data, 3)
        if self.max_order > 3:
            self.m4 = self.increment_moments_mean(self.m4, np_data, 4)

        # update the thetas
        if self.iteration > 1:
            if self.max_order > 2:
                self.theta3 = self.m3 - 3 * self.m1 * self.m2 + 2 * np.power(self.m1, 3)
            if self.max_order > 3:
                self.theta4 = (
                    self.m4 - 4 * self.m1 * self.m3 + 6 * np.square(self.m1) * self.m2
                    - 3 * np.power(self.m1, 4)
                )

    def increment_moments_mean(
        self, moment: npt.NDArray[np.float_], np_data, power
    ) -> npt.NDArray[np.float_]:
        """
        This function computes the iterative mean of the given moment
        """
        return moment + (np.power(np_data, power) - moment) / self.iteration

    def get_mean(self) -> npt.NDArray[np.float_]:
        return self.m1

    def get_variance(self) -> npt.NDArray[np.float_]:
        return self.theta2

    def get_skewness(self) -> npt.NDArray[np.float_]:
        epsilon: float = 1e-12
        n_idx = np.where(abs(self.theta2) > epsilon)
        skewness: npt.NDArray[np.float_] = np.zeros(len(self.theta2))
        skewness[n_idx] = self.theta3[n_idx] / np.power(self.theta2[n_idx], 1.5)
        return skewness

    def get_kurtosis(self) -> npt.NDArray[np.float_]:
        # FIXME: kurtosis is not matching scipy.stats.kurtosis
        epsilon: float = 1e-12
        n_idx = np.where(abs(self.theta3) > epsilon)
        kurtosis: npt.NDArray[np.float_] = np.zeros(len(self.theta3))
        kurtosis[n_idx] = self.theta4[n_idx] / np.power(self.theta3[n_idx], 2)
        return kurtosis

    def save_state(self):
        """
        This function saves the state of the moments
        """
        mean = self.iterative_mean.save_state()
        return {"mean": mean, "m1": self.m1,
                "m2": self.m2, "m3": self.m3, "m4": self.m4,
                "theta2": self.theta2, "theta3": self.theta3,
                "theta4": self.theta4, "increment": self.iteration}

    def load_from_state(self, state: dict):

        self.iterative_stat.load_from_state(state["mean"])
        self.m1 = state["m1"]
        self.m2 = state["m2"]
        self.m3 = state["m3"]
        self.m4 = state["m4"]
        self.theta2 = state["theta2"]
        self.theta3 = state["theta3"]
        self.theta4 = state["theta4"]
        self.iteration = state["increment"]
