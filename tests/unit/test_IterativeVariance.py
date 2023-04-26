import unittest
import numpy as np


from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.utils.logger import logger


class TestIterativeVariance(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        gt_stats = np.var(sample,ddof=1) #compute the unbiased estimatorof the variance
        iterative_stats = IterativeVariance(dim = 1)
        res = []
        for x in sample :
            res.append(x)
            iterative_stats.increment(x)    
        logger.info(f'{iterative_stats.get_stats()}')     
        self.assertAlmostEqual(gt_stats, iterative_stats.get_stats()[0], delta=10e-2)
        self.assertEqual(iterative_stats.get_stats().shape, (1,))

    def test_increment_multidim(self):
        sample = np.array([[10.0, 11.0, 12.0], [1000.0, 1001.0, 1002.0]])
        sample = np.transpose(sample)
        gt_stats = np.var(sample,ddof=1, axis=0) #compute the unbiased estimatorof the variance
        iterative_stats = IterativeVariance(dim = 2)
        res = []
        for x in sample :
            res.append(x)
            iterative_stats.increment(x) 
        logger.info(f'{iterative_stats.get_stats()}')       
        res = np.allclose(gt_stats,iterative_stats.get_stats(), atol=10e-10)
        self.assertTrue(res)
        self.assertEqual(iterative_stats.get_stats().shape, (2,))