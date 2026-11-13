import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)

logger = logging.getLogger(__name__)


class IterativeMoments(AbstractIterativeStatistics):
    """
    Compute higher-order central moments iteratively.

    Based on: https://arxiv.org/pdf/1510.04923.pdf
    "Simpler Online Updates for Arbitrary-Order Central Moments"

    Parameters
    ----------
    dim : int
        Dimensionality of data.
    max_order : int
        Maximum moment order (1-4).
    state : StateDict, optional
        Previous state to restore.
    """

    def __init__(self, dim: int, max_order: int, state: StateDict | None = None) -> None:
        if not isinstance(max_order, int) or max_order < 1 or max_order > 4:
            raise ValueError("max_order must be an integer between 1 and 4")

        super().__init__(dim, state)
        self.max_order: int = max_order

        # Cascading moments depending on requested max_order
        self.m1: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)

        if self.max_order > 1:
            self.m2: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)
            self.theta2: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)

        if self.max_order > 2:
            self.m3: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)
            self.theta3: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)

        if self.max_order > 3:
            self.m4: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)
            self.theta4: NDArray[np.float64] = np.zeros(self.dimension, dtype=np.float64)

    def increment(self, data: NDArray[np.float64]) -> None:
        """
        Update moments with new data point.

        Parameters
        ----------
        data : NDArray[np.float64]
            New data point.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.shape != self.m1.shape:
            raise ValueError(f"Expected shape {self.m1.shape}, got {data.shape}")

        self.iteration += 1
        delta = data - self.m1
        delta_n = delta / self.iteration
        self.m1 += delta_n

        if self.max_order > 1:
            self.m2 += delta * (delta - delta_n)

        if self.max_order > 2:
            delta_2 = delta * delta
            delta_n_2 = delta_n * delta_n
            self.m3 += -3.0 * delta_n * self.m2 + delta * (delta_2 - delta_n_2)

        if self.max_order > 3:
            delta_n_2 = delta_n * delta_n
            self.m4 += (
                -4.0 * delta_n * self.m3
                - 6.0 * delta_n_2 * self.m2
                + delta * (delta * delta * delta - delta_n * delta_n_2)
            )

    def get_mean(self) -> NDArray[np.float64]:
        """Return first moment (mean)."""
        return self.m1

    def get_variance(self) -> NDArray[np.float64]:
        """Return second moment (variance)."""
        if self.max_order < 2:
            raise ValueError("max_order must be >= 2 for variance")
        return self.m2 / self.iteration

    def get_skewness(self) -> NDArray[np.float64]:
        """Return standardized third moment (skewness)."""
        if self.max_order < 3:
            raise ValueError("max_order must be >= 3 for skewness")
        return np.sqrt(self.iteration) * self.m3 / np.power(self.m2, 1.5)

    def get_kurtosis(self) -> NDArray[np.float64]:
        """Return standardized fourth moment (kurtosis)."""
        if self.max_order < 4:
            raise ValueError("max_order must be >= 4 for kurtosis")
        return self.iteration * self.m4 / (self.m2**2)

    def save_state(self) -> StateDict:
        """Save state of all moments."""
        state = super().save_state()
        state["m1"] = self.m1.copy()
        state["m2"] = self.m2.copy() if self.max_order > 1 else None
        state["m3"] = self.m3.copy() if self.max_order > 2 else None
        state["m4"] = self.m4.copy() if self.max_order > 3 else None
        state["max_order"] = self.max_order
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of moments."""
        super().load_from_state(state)
        self.m1 = state["m1"]
        if self.max_order > 1:
            self.m2 = state["m2"]
        if self.max_order > 2:
            self.m3 = state["m3"]
        if self.max_order > 3:
            self.m4 = state["m4"]
