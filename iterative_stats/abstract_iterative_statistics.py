from abc import ABC, abstractmethod
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class StateDict(TypedDict, total=False):
    """Type definition for saved state."""

    iteration: int
    state: NDArray[np.float64]


class AbstractIterativeStatistics(ABC):
    """
    Base class for computing iterative statistics.

    Parameters
    ----------
    dim : int
        Dimensionality of the data.
    state : StateDict, optional
        Previous state to restore from.
    """

    def __init__(self, dim: int, state: StateDict | None = None) -> None:
        self.dimension = dim
        if state is None:
            self.state: NDArray[np.float64] = np.zeros(dim, dtype=np.float64)
            self.iteration = 0
        else:
            self.load_from_state(state)

    @abstractmethod
    def increment(self, data: NDArray[np.float64]) -> None:
        """
        Update statistics with new data point.

        Parameters
        ----------
        data : NDArray[np.float64]
            Input data matching the dimension.
        """

    def save_state(self) -> StateDict:
        """Save current state for later restoration."""
        return {"iteration": self.iteration, "state": self.state.copy()}

    def load_from_state(self, state: StateDict) -> None:
        """
        Restore state from saved dictionary.

        Parameters
        ----------
        state : StateDict
            Dictionary with 'iteration' and 'state' keys.
        """
        self.iteration = state.get("iteration", 0)
        self.state = state.get("state", np.zeros(self.dimension, dtype=np.float64))

    def get_stats(self) -> NDArray[np.float64]:
        """Return current state."""
        return self.state

    def get_iteration(self) -> int:
        """Return current iteration count."""
        return self.iteration
