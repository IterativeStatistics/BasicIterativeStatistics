import unittest
import numpy as np



from iterative_stats.iterative_threshold import IterativeThreshold
from iterative_stats.utils.logger import logger


class TestAbstractIterativeThreshold(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        iterative_stats = IterativeThreshold(dim = 1, min_threshold= 1., max_threshold= 11.)
        gt = ( (sample <= 11.) & (sample >= 1.)).sum()
        for x in sample :
            iterative_stats.increment(x)
        logger.info(f' res={iterative_stats.get_stats()}, gt= {gt}')
        self.assertAlmostEqual(gt, iterative_stats.get_stats(), delta=10e-2)


    def test_increment_multidim(self): 
        delta = 0.1
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)

        min_threshold = np.array([1,-2])
        max_threshold = np.array([10,15])
        iterative_stats = IterativeThreshold(dim = 2, min_threshold= min_threshold, max_threshold= max_threshold)
        
        gt = ((sample <= max_threshold) & (sample >= min_threshold)).sum(axis=0)

        for x in sample :
            iterative_stats.increment(x)

        logger.info(f' res={iterative_stats.get_stats()}, gt= {gt}')

        res = np.allclose(gt,iterative_stats.get_stats(), atol=10e-10)
        self.assertTrue(res)

