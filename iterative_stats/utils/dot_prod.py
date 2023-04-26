import numpy as np

from iterative_stats.utils.logger import logger

def multi_dim_dotproduct(data_1: np.array, data_2: np.array, dim: int = 1):
    if dim == 1 :
        return np.dot(data_1, data_2)
    return np.array([np.dot(data_1[:,k], data_2[:,k]) for k in range(dim)])
