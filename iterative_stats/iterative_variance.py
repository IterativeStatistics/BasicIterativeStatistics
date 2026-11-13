import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)
from iterative_stats.iterative_mean import IterativeMean

logger = logging.getLogger(__name__)


class IterativeVariance(AbstractIterativeStatistics):
    """Iterative computation of variance using Welford's algorithm."""

    def __init__(self, dim: int = 1, state: StateDict | None = None) -> None:
        if state is None:
            self.mean = IterativeMean(dim)
            self.sumOfCenteredSquares: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
        super().__init__(dim, state)

    def increment(self, data: NDArray[np.float64] | float) -> None:
        """
        Update variance with new data point.

        Parameters
        ----------
        data : NDArray[np.float64] | float
            New data point.
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))

        self.iteration += 1
        if self.iteration > 1:
            mean_data = data - self.mean.get_stats()
            weight = (self.iteration - 1) / self.iteration
            self.sumOfCenteredSquares += weight * (mean_data**2)

        # update mean
        self.mean.increment(data)

        # compute variance
        if self.iteration > 1:
            self.state = self.sumOfCenteredSquares / (self.iteration - 1)
        logger.debug(f"iteration={self.iteration}, variance={self.state}")

    def get_variance(self) -> NDArray[np.float64]:
        """Return current variance estimate."""
        return self.state

    def get_mean(self) -> NDArray[np.float64]:
        """Return current mean estimate."""
        return self.mean.get_stats()

    def save_state(self) -> StateDict:
        """Save state of variance and internal mean."""
        state = super().save_state()
        state["mean"] = self.mean.save_state()
        state["sumOfCenteredSquares"] = self.sumOfCenteredSquares.copy()
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of variance and internal mean."""
        super().load_from_state(state)
        self.mean = IterativeMean(self.dimension, state.get("mean"))
        self.sumOfCenteredSquares = state.get("sumOfCenteredSquares", np.zeros(self.dimension, dtype=np.float64))
