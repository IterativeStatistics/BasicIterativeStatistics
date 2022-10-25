import unittest
import numpy as np


from src.iterative_variance import IterativeVariance
from src.utils.logger import logger


class TestIterativeVariance(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        gt_stats = np.var(sample,ddof=1) #compute the unbiased estimatorof the variance
        iterative_stats = IterativeVariance({'vector_size' : 1})
        res = []
        for x in sample :
            res.append(x)
            iterative_stats.increment(x)        
        self.assertAlmostEqual(gt_stats, iterative_stats.get_stats()[0], delta=10e-2)