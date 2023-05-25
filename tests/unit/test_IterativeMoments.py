import unittest
import numpy as np


from iterative_stats.iterative_moments import IterativeMoments
from iterative_stats.utils.logger import logger
import scipy.stats as stats


class TestIterativeMoments(unittest.TestCase):

    def test_increment(self):
        # sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
        # set seed for reproducibility
        np.random.seed(1234)
        # generate 100 random numbers from normal distribution
        sample = np.random.normal(0, 1, 40)
        mean = np.mean(sample)
        variance = np.var(sample,ddof=1) #compute the unbiased estimatorof the variance
        skewness = stats.skew(sample, bias=False)
        kurtosis = stats.kurtosis(sample, bias=False)

        iterative_stats = IterativeMoments(4, dim = 1)
        res = []
        for x in sample :
            res.append(x)
            iterative_stats.increment(x)    
        logger.info(f'{iterative_stats.get_stats()}')
        self.assertAlmostEqual(mean, iterative_stats.get_mean()[0], delta=10e-2)
        self.assertAlmostEqual(variance, iterative_stats.get_variance()[0], delta=10e-2)
        self.assertAlmostEqual(skewness, iterative_stats.get_skewness()[0], delta=10e-2)
        # FIXME: Kurtosis is incorrect - needs someone to check the equations
        self.assertAlmostEqual(kurtosis, iterative_stats.get_kurtosis()[0], delta=10e-2)
        self.assertEqual(iterative_stats.get_stats().shape, (1,))
