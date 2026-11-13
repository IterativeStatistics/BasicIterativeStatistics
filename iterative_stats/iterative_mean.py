import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)

logger = logging.getLogger(__name__)


class IterativeMean(AbstractIterativeStatistics):
    """Iterative computation of mean using Welford's algorithm."""

    def increment(self, data: NDArray[np.float64] | float) -> None:
        """
        Update mean with new data point.

        Parameters
        ----------
        data : NDArray[np.float64] | float
            New data point(s).
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))
        if data.shape != self.state.shape:
            raise ValueError(f"Expected shape {self.state.shape}, got {data.shape}")

        self.iteration += 1
        self.state += (data - self.state) / self.iteration
        logger.debug(f"iteration={self.iteration}, mean={self.state}")


class IterativeShiftedMean(AbstractIterativeStatistics):
    """Iterative mean with shift parameter."""

    def __init__(self, dim: int = 1, state: StateDict | None = None) -> None:
        self.previous_shift: NDArray[np.float64] | None = None
        super().__init__(dim, state)

    def increment(self, data: NDArray[np.float64] | float, shift: NDArray[np.float64] | float) -> None:
        """
        Update shifted mean.

        Parameters
        ----------
        data : NDArray[np.float64] | float
            New data point.
        shift : NDArray[np.float64] | float
            Shift parameter.
        """
        data = np.atleast_1d(np.asarray(data, dtype=np.float64))
        shift = np.atleast_1d(np.asarray(shift, dtype=np.float64))

        self.iteration += 1
        if self.iteration == 1:
            self.state = data - shift
        else:
            self.state *= 1.0 - 1.0 / self.iteration
            self.state += (data - self.previous_shift) / self.iteration
            self.state += self.previous_shift - shift

        self.previous_shift = shift.copy()

    def save_state(self) -> StateDict:
        """Save state including previous shift."""
        state = super().save_state()
        state["previous_shift"] = self.previous_shift
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state including previous shift."""
        super().load_from_state(state)
        self.previous_shift = state.get("previous_shift")
