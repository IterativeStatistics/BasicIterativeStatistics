# Iterative quantile estimation by use of a Robbins-Monro algorithm
# Implementer: Alejandro Ribés

import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)

logger = logging.getLogger(__name__)


class IterativeQuantile(AbstractIterativeStatistics):
    """
    Iterative quantile estimation using Robbins-Monro algorithm.

    Parameters
    ----------
    dim : int
        Dimensionality of the data.
    state : StateDict, optional
        Previous state to restore.
    alpha : float, optional
        Target quantile level (0 to 1). Default is 0.5 (median).
    max_it : int, optional
        Maximum number of iterations.
    """

    def __init__(
        self,
        dim: int = 1,
        state: StateDict | None = None,
        alpha: float = 0.5,
        max_it: int = 0,
    ) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        if state is None:
            self.state_min: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
            self.state_max: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
            self.indicator: NDArray[np.int8] = np.zeros(dim, dtype=np.int8)
            self.C: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)

        super().__init__(dim, state)

        self.max_it = max_it
        self.alpha = alpha

    def increment(self, data: NDArray[np.float64]) -> None:
        """
        Update quantile estimate with new data point.

        Parameters
        ----------
        data : NDArray[np.float64]
            New data point.
        """
        data = np.asarray(data, dtype=np.float64)
        self.iteration += 1

        if self.iteration == 1:
            self.state_min = data.copy()
            self.state_max = data.copy()
        else:
            self.state_min = np.minimum(data, self.state_min)
            self.state_max = np.maximum(data, self.state_max)

        if self.max_it <= 1.0:
            logger.error(f"MaxIterations={self.max_it}, has not been set (or set to <= 1)")

        gamma = 0.5 + 0.5 * ((self.iteration - 1.0) / (self.max_it - 1.0))
        C = np.abs(self.state_max - self.state_min)

        self.indicator = (data <= self.state).astype(np.int8)

        if self.iteration == 1:
            self.state = data.copy()
        else:
            self.state -= (C / (self.iteration**gamma)) * (self.indicator - self.alpha)

    def save_state(self) -> StateDict:
        """Save state of quantile estimation."""
        state = super().save_state()
        state["indicator"] = self.indicator.copy()
        state["C"] = self.C.copy()
        state["state_min"] = self.state_min.copy()
        state["state_max"] = self.state_max.copy()
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of quantile estimation."""
        super().load_from_state(state)
        self.state_min = state.get("state_min", np.zeros(self.dimension, dtype=np.float64))
        self.state_max = state.get("state_max", np.zeros(self.dimension, dtype=np.float64))
        self.indicator = state.get("indicator", np.zeros(self.dimension, dtype=np.int8))
        self.C = state.get("C", np.zeros(self.dimension, dtype=np.float64))
