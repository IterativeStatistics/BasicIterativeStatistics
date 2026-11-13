import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)
from iterative_stats.iterative_mean import IterativeMean

logger = logging.getLogger(__name__)


class IterativeCovariance(AbstractIterativeStatistics):
    """Iterative computation of covariance between two data streams."""

    def __init__(self, dim: int = 1, state: StateDict | None = None) -> None:
        self.mean_1 = IterativeMean(dim)
        self.mean_2 = IterativeMean(dim)
        super().__init__(dim, state)

    def increment(self, data_1: NDArray[np.float64], data_2: NDArray[np.float64]) -> None:
        """
        Update covariance with new data pair.

        Parameters
        ----------
        data_1 : NDArray[np.float64]
            First data point.
        data_2 : NDArray[np.float64]
            Second data point.
        """
        data_1 = np.asarray(data_1, dtype=np.float64)
        data_2 = np.asarray(data_2, dtype=np.float64)

        # update mean
        self.iteration += 1
        prev_mean_1 = self.mean_1.get_stats().copy()
        prev_mean_2 = self.mean_2.get_stats().copy()
        self.mean_1.increment(data_1)
        self.mean_2.increment(data_2)

        # update covariance
        if self.iteration > 1:
            self.state *= self.iteration - 2
            x = np.multiply(data_1 - prev_mean_1, data_2 - prev_mean_2)
            diff_mean = np.multiply(
                prev_mean_1 - self.mean_1.get_stats(),
                prev_mean_2 - self.mean_2.get_stats(),
            )
            self.state = self.state + x - diff_mean * self.iteration
            self.state = self.state / (self.iteration - 1)

    def get_covariance(self) -> NDArray[np.float64]:
        """Return current covariance estimate."""
        return self.state

    def get_mean_1(self) -> NDArray[np.float64]:
        """Return mean of first data stream."""
        return self.mean_1.get_stats()

    def get_mean_2(self) -> NDArray[np.float64]:
        """Return mean of second data stream."""
        return self.mean_2.get_stats()

    # Backward compatibility aliases
    def get_mean1(self) -> NDArray[np.float64]:
        """Deprecated: use get_mean_1() instead."""
        return self.get_mean_1()

    def get_mean2(self) -> NDArray[np.float64]:
        """Deprecated: use get_mean_2() instead."""
        return self.get_mean_2()

    def getCovariance(self) -> NDArray[np.float64]:
        """Deprecated: use get_covariance() instead."""
        return self.get_covariance()

    def save_state(self) -> StateDict:
        """Save state of covariance and internal means."""
        state = super().save_state()
        state["mean_1"] = self.mean_1.save_state()
        state["mean_2"] = self.mean_2.save_state()
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of covariance and internal means."""
        super().load_from_state(state)
        self.mean_1 = IterativeMean(self.dimension, state=state.get("mean_1"))
        self.mean_2 = IterativeMean(self.dimension, state=state.get("mean_2"))
