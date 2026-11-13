import numpy as np
from numpy.typing import NDArray


def multi_dim_dotproduct(
    data_1: NDArray[np.float64], data_2: NDArray[np.float64], dim: int = 1
) -> NDArray[np.float64] | float:
    """
    Compute dot products for multi-dimensional arrays.

    Parameters
    ----------
    data_1 : NDArray[np.float64]
        First input array.
    data_2 : NDArray[np.float64]
        Second input array.
    dim : int, optional
        Number of dimensions (default: 1).

    Returns
    -------
    NDArray[np.float64] | float
        Dot product result. Returns scalar for dim=1, array otherwise.
    """
    if dim == 1:
        # For scalar case: compute standard dot product
        # Handle both 1D and 2D (Nx1) array shapes
        d1_flat = data_1.ravel() if data_1.ndim > 1 else data_1
        d2_flat = data_2.ravel() if data_2.ndim > 1 else data_2
        return np.dot(d1_flat, d2_flat)

    # Sum over samples, returning one dot product per data dimension.
    return np.einsum("ik,ik->k", data_1, data_2)
