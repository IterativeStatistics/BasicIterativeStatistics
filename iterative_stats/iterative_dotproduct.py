import logging

import numpy as np
from numpy.typing import NDArray

from iterative_stats.abstract_iterative_statistics import (
    AbstractIterativeStatistics,
    StateDict,
)
from iterative_stats.iterative_mean import IterativeShiftedMean
from iterative_stats.utils.dot_prod import multi_dim_dotproduct

logger = logging.getLogger(__name__)


class IterativeDotProduct(AbstractIterativeStatistics):
    """
    Iterative dot product computation with shift.

    Computes: sum_{k=0}^{N} (A_k - shift_N)(B_k - shift_N) / (N-1)

    Parameters
    ----------
    dim : int
        Dimensionality of the data.
    iterative_shifted_mean_1 : IterativeShiftedMean, optional
        External shifted mean for first data stream.
    iterative_shifted_mean_2 : IterativeShiftedMean, optional
        External shifted mean for second data stream.
    state : StateDict, optional
        Previous state to restore.
    """

    def __init__(
        self,
        dim: int = 1,
        iterative_shifted_mean_1: IterativeShiftedMean | None = None,
        iterative_shifted_mean_2: IterativeShiftedMean | None = None,
        state: StateDict | None = None,
    ) -> None:
        self.previous_shift: NDArray[np.float64] | None = None

        if iterative_shifted_mean_1 is None:
            self.iterative_shifted_mean_1 = IterativeShiftedMean(dim)
            self.external_mean_1 = False
        else:
            self.iterative_shifted_mean_1 = iterative_shifted_mean_1
            self.external_mean_1 = True

        if iterative_shifted_mean_2 is None:
            self.iterative_shifted_mean_2 = IterativeShiftedMean(dim)
            self.external_mean_2 = False
        else:
            self.iterative_shifted_mean_2 = iterative_shifted_mean_2
            self.external_mean_2 = True

        super().__init__(dim, state)
        self.data_1: NDArray[np.float64] | None = None
        self.data_2: NDArray[np.float64] | None = None

    def increment(
        self,
        data_1: NDArray[np.float64],
        data_2: NDArray[np.float64],
        shift: NDArray[np.float64],
    ) -> None:
        """
        Update dot product with new data pair.

        Parameters
        ----------
        data_1 : NDArray[np.float64]
            First data point.
        data_2 : NDArray[np.float64]
            Second data point.
        shift : NDArray[np.float64]
            Shift parameter.
        """
        data_1 = np.asarray(data_1, dtype=np.float64)
        data_2 = np.asarray(data_2, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)

        self.iteration += 1

        if self.iteration == 1:
            self.data_1 = data_1.copy()
            self.data_2 = data_2.copy()
        elif self.iteration == 2:
            # Create consistent 2D arrays
            self.data_1 = np.vstack((self.data_1, data_1))
            self.data_2 = np.vstack((self.data_2, data_2))
            self.state = multi_dim_dotproduct(self.data_1 - shift, self.data_2 - shift, self.dimension)
            del self.data_1, self.data_2
        else:
            diff_shift = self.previous_shift - shift
            self.state *= 1.0 - 1.0 / (self.iteration - 1)
            self.state += np.multiply(data_1 - self.previous_shift, data_2 - self.previous_shift) / (self.iteration - 1)
            val = (
                self.iterative_shifted_mean_1.get_stats()
                + self.iterative_shifted_mean_2.get_stats()
                + diff_shift
                + (data_1 + data_2 - shift - self.previous_shift) / (self.iteration - 1)
            )
            self.state += val * diff_shift

        # Update iterative shifted mean if not external
        if not self.external_mean_1:
            self.iterative_shifted_mean_1.increment(data_1, shift)

        if not self.external_mean_2:
            self.iterative_shifted_mean_2.increment(data_2, shift)

        self.previous_shift = shift.copy()

    def get_mean_1(self) -> NDArray[np.float64]:
        """Return mean of first data stream."""
        return self.iterative_shifted_mean_1.get_stats()

    def get_mean_2(self) -> NDArray[np.float64]:
        """Return mean of second data stream."""
        return self.iterative_shifted_mean_2.get_stats()

    def save_state(self) -> StateDict:
        """Save state of dot product and internal means."""
        state = super().save_state()
        state["previous_shift"] = self.previous_shift.copy() if self.previous_shift is not None else None
        if not self.external_mean_1:
            state["external_mean_1"] = self.iterative_shifted_mean_1.save_state()
        if not self.external_mean_2:
            state["external_mean_2"] = self.iterative_shifted_mean_2.save_state()
        return state

    def load_from_state(self, state: StateDict) -> None:
        """Restore state of dot product and internal means."""
        super().load_from_state(state)
        self.previous_shift = state.get("previous_shift")
        if state.get("external_mean_1") is not None:
            self.external_mean_1 = False
            self.iterative_shifted_mean_1 = IterativeShiftedMean(self.dimension, state.get("external_mean_1"))
        if state.get("external_mean_2") is not None:
            self.external_mean_2 = False
            self.iterative_shifted_mean_2 = IterativeShiftedMean(self.dimension, state.get("external_mean_2"))
