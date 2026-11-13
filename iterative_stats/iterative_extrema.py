import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)

logger = logging.getLogger(__name__)


class IterativeExtrema(AbstractIterativeStatistics):
    """Iterative computation of minimum and maximum values."""

    def __init__(self, dim: int = 1, state: StateDict | None = None) -> None:
        self.state_min: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
        super().__init__(dim, state)

    def increment(self, data: NDArray[np.float64]) -> None:
        """
        Update extrema with new data point.

        Parameters
        ----------
        data : NDArray[np.float64]
            New data point.
        """
        data = np.asarray(data, dtype=np.float64)
        self.state_min = np.minimum(data, self.state_min)
        self.state = np.maximum(data, self.state)

    def get_min(self) -> NDArray[np.float64]:
        """Return current minimum values."""
        return self.state_min

    def get_max(self) -> NDArray[np.float64]:
        """Return current maximum values."""
        return self.state

    def save_state(self) -> StateDict:
        """Save state of extrema."""
        state = super().save_state()
        state["state_min"] = self.state_min.copy()
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of extrema."""
        super().load_from_state(state)
        self.state_min = state.get("state_min", np.zeros(self.dimension, dtype=np.float64))
