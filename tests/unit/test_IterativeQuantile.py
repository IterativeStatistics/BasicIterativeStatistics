import unittest
import numpy as np

from iterative_stats.iterative_quantile import IterativeQuantile
from iterative_stats.utils.logger import logger 

class TestQuantileIterativeStatistics(unittest.TestCase):
    
    def test_increment_iterativeMedian_NormalDistribution(self):
        mu = 20
        sigma = 10 
        sample = np.random.normal(mu, sigma, 5000)
        me = np.median(sample)
 
        iterativeMedian = IterativeQuantile(vector_size = 1)
        iterativeMedian.setDesiredQuantile(0.5)
        
        iterativeMedian.setMaxIterations(sample.size)
        for x in sample :
            iterativeMedian.increment(x)
        self.assertAlmostEqual(me, iterativeMedian.get_stats()[0], delta=0.5)


    def test_increment_iterativeQuantile_NormalDistribution(self):
        mu = 20
        sigma = 10 
        sample = np.random.normal(mu, sigma, 5000)
        qO7 = np.quantile(sample, 0.7)
 
        iterativeQ07 = IterativeQuantile(vector_size = 1)
        iterativeQ07.setDesiredQuantile(0.7)
        

        iterativeQ07.setMaxIterations(sample.size)
        for x in sample :
            iterativeQ07.increment(x)
        self.assertAlmostEqual(me, iterativeQ07.get_stats()[0], delta=0.5)
 


