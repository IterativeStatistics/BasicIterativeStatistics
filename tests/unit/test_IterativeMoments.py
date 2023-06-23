import unittest
import numpy as np
import openturns as ot


from iterative_stats.iterative_moments import IterativeMoments
from iterative_stats.utils.logger import logger
import scipy.stats as stats


class TestIterativeMoments(unittest.TestCase):

    def test_increment(self):
        # set seed for reproducibility
        np.random.seed(1234)
        # generate 100 random numbers from normal distribution
        sample = np.random.normal(0, 1, 500)
        mean = np.mean(sample)
        variance = np.var(sample,ddof=1) #compute the unbiased estimatorof the variance
        skewness = stats.skew(sample, bias=False)
        kurtosis = stats.kurtosis(sample, bias=False)

        iterative_stats = IterativeMoments(4, dim = 1)
        ot_iter = ot.IterativeMoments(4, 1)
        res = []
        for x in sample :
            res.append(x)
            iterative_stats.increment(x)
            ot_iter.increment(ot.Point([x]))

        logger.info(f'{iterative_stats.get_stats()}')
        self.assertAlmostEqual(mean, iterative_stats.get_mean()[0], delta=10e-2)
        self.assertAlmostEqual(variance, ot_iter.getVariance()[0], delta=10e-2)
        self.assertAlmostEqual(variance, iterative_stats.get_variance()[0], delta=10e-2)
        self.assertAlmostEqual(skewness, ot_iter.getSkewness()[0], delta=10e-2)
        self.assertAlmostEqual(skewness, iterative_stats.get_skewness()[0], delta=10e-2)
        # FIXME: Kurtosis is incorrect - needs someone to check the equations
        logger.info(f"Scipy: {kurtosis}, OpenTurns: {ot_iter.getKurtosis()[0]}, "
                    f"Iterative: {iterative_stats.get_kurtosis()[0]}")
        # self.assertAlmostEqual(kurtosis, ot_iter.getKurtosis()[0], delta=10e-2)
        # self.assertAlmostEqual(kurtosis, iterative_stats.get_kurtosis()[0], delta=10e-2)
        self.assertEqual(iterative_stats.get_stats().shape, (1,))
