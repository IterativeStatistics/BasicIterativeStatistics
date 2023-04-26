import unittest
import numpy as np



from iterative_stats.iterative_extrema import IterativeExtrema
from iterative_stats.utils.logger import logger


class TestAbstractIterativeStatistics(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        gt_min = np.min(sample)
        gt_max = np.max(sample)
        iterative_stats = IterativeExtrema(dim = 1)
        for x in sample :
            iterative_stats.increment(x)
        logger.info(f' min={iterative_stats.get_min()[0]}, max= {iterative_stats.get_max()[0]}')
        self.assertAlmostEqual(gt_min, iterative_stats.get_min()[0], delta=10e-2)
        self.assertAlmostEqual(gt_max, iterative_stats.get_max()[0], delta=10e-2)

    def test_increment_multidim(self):
        delta = 1.e-10 
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)
        gt_min = np.min(sample, axis=0)
        gt_max = np.max(sample, axis=0)
        iterative_stats = IterativeExtrema(dim = 2)
        for x in sample :
            iterative_stats.increment(x)
        logger.info(f' min={iterative_stats.get_min()}, max= {iterative_stats.get_max()[0]}')
        res = np.allclose(gt_min,iterative_stats.get_min(), atol=10e-10)
        self.assertTrue(res)

        res = np.allclose(gt_max,iterative_stats.get_max(), atol=10e-10)
        self.assertTrue(res)