import unittest
import numpy as np



from iterative_stats.iterative_mean import IterativeMean



class TestAbstractIterativeStatistics(unittest.TestCase):
      
    def test_increment(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        mu = np.mean(sample)
        iterativeMean = IterativeMean({'vector_size' : 1})
        for x in sample :
            iterativeMean.increment(x)
        self.assertAlmostEqual(mu, iterativeMean.get_stats()[0], delta=10e-2)