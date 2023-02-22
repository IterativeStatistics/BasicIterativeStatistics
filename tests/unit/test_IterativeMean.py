import unittest
import numpy as np

from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_mean import IterativeShiftedMean

from iterative_stats.utils.logger import logger 

class TestMeanIterativeStatistics(unittest.TestCase):
      
    def test_increment_iterativeMean(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        mu = np.mean(sample)
        iterativeMean = IterativeMean(vector_size = 1)
        for x in sample :
            iterativeMean.increment(x)
        self.assertAlmostEqual(mu, iterativeMean.get_stats()[0], delta=10e-2)

    def test_increment_iterativeShiftedMean(self):
        sample =  np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        shifted_sample = np.array([-1., 5., -6., 13.0, -16.0, -12.2])
        iterativeMean = IterativeShiftedMean(vector_size = 1)

        cpt = 0
        for x, shift in zip(sample, shifted_sample) :
            iterativeMean.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt])
            logger.info(f'mu: {mu}, other={iterativeMean.get_stats()}')
            cpt += 1
            self.assertAlmostEqual(mu, iterativeMean.get_stats(), delta=10e-2)

