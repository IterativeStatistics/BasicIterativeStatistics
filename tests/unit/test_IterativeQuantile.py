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
 
        iterativeMedian = IterativeQuantile(dim = 1, alpha=0.5, max_it=sample.size)

        for x in sample :
            iterativeMedian.increment(x)
        
        logger.info(f'me: {me} / {iterativeMedian.get_stats()}')
        self.assertAlmostEqual(me, iterativeMedian.get_stats(), delta=0.5)


    def test_increment_iterativeQuantile_NormalDistribution(self):
        mu = 20
        sigma = 10 
        sample = np.random.normal(mu, sigma, 5000)
        qO7 = np.quantile(sample, 0.7)
 
        iterativeQ07 = IterativeQuantile(dim = 1, alpha=0.7, max_it=sample.size)
        
        for x in sample :
            iterativeQ07.increment(x)
        logger.info(f'qO7: {qO7} / {iterativeQ07.get_stats()}')
        self.assertAlmostEqual(qO7, iterativeQ07.get_stats(), delta=0.5)
 
    # def test_increment_iterativeQuantile_NormalDistribution_ft(self):
    #     mu = 20
    #     sigma = 10 
    #     sample = np.random.normal(mu, sigma, 100)
    #     qO7 = np.quantile(sample, 0.7)
 
    #     iterativeQ07 = IterativeQuantile(dim = 1, alpha=0.7, max_it=sample.size)

    #     for x in sample[:10] :
    #         iterativeQ07.increment(x)

    #     state = iterativeQ07.save_state()
    #     iterativeQ07_ft = IterativeQuantile(dim = 1, alpha=0.7, max_it=sample.size, state = state)

    #     for x in sample[10:] :
    #         iterativeQ07_ft.increment(x)

    #     logger.info(f'qO7: {qO7} / {iterativeQ07_ft.get_stats()}')
    #     self.assertAlmostEqual(qO7, iterativeQ07_ft.get_stats(), delta=1)
 

