import unittest
import numpy as np



from iterative_stats.iterative_extrema import IterativeExtrema
from iterative_stats.utils.logger import logger


class TestAbstractIterativeStatistics(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        gt_min = np.min(sample)
        gt_max = np.max(sample)
        iterativeMean = IterativeExtrema(vector_size = 1)
        for x in sample :
            iterativeMean.increment(x)
        logger.info(f' min={iterativeMean.get_min()[0]}, max= {iterativeMean.get_max()[0]}')
        self.assertAlmostEqual(gt_min, iterativeMean.get_min()[0], delta=10e-2)
        self.assertAlmostEqual(gt_max, iterativeMean.get_max()[0], delta=10e-2)