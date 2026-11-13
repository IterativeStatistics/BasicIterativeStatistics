import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)

logger = logging.getLogger(__name__)


class IterativeThreshold(AbstractIterativeStatistics):
    """
    Iterative threshold counter.

    Counts how many data points fall within a specified range.

    Parameters
    ----------
    dim : int
        Dimensionality of the data.
    min_threshold : NDArray[np.float64], optional
        Minimum threshold value.
    max_threshold : NDArray[np.float64], optional
        Maximum threshold value.
    state : StateDict, optional
        Previous state to restore.
    """

    def __init__(
        self,
        dim: int = 1,
        min_threshold: NDArray[np.float64] | None = None,
        max_threshold: NDArray[np.float64] | None = None,
        state: StateDict | None = None,
    ) -> None:
        self.min_threshold = np.asarray(min_threshold, dtype=np.float64) if min_threshold is not None else None
        self.max_threshold = np.asarray(max_threshold, dtype=np.float64) if max_threshold is not None else None
        super().__init__(dim, state)

    def increment(self, data: NDArray[np.float64]) -> None:
        """
        Update threshold counter with new data point.

        Parameters
        ----------
        data : NDArray[np.float64]
            New data point.
        """
        data = np.asarray(data, dtype=np.float64)
        self.iteration += 1

        if self.min_threshold is not None:
            res_min = data >= self.min_threshold
        else:
            res_min = np.ones(self.dimension, dtype=bool)

        if self.max_threshold is not None:
            res_max = data <= self.max_threshold
        else:
            res_max = np.ones(self.dimension, dtype=bool)

        self.state += (res_min & res_max).astype(np.float64)

    def save_state(self) -> StateDict:
        """Save threshold configuration along with the accumulated count."""
        state = super().save_state()
        state["min_threshold"] = self.min_threshold.copy() if self.min_threshold is not None else None
        state["max_threshold"] = self.max_threshold.copy() if self.max_threshold is not None else None
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore threshold configuration along with the accumulated count."""
        super().load_from_state(state)
        min_threshold = state.get("min_threshold")
        max_threshold = state.get("max_threshold")
        self.min_threshold = np.asarray(min_threshold, dtype=np.float64) if min_threshold is not None else None
        self.max_threshold = np.asarray(max_threshold, dtype=np.float64) if max_threshold is not None else None
