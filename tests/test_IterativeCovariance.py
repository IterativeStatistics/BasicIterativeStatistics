import unittest
import numpy as np



from src.iterative_covariance import IterativeCovariance
from src.utils.logger import logger

class TestCov:
    def __init__(self, sample_1, sample_2):
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.iterativeCov = IterativeCovariance({'vector_size' :1})
    
    def run(self):
        sample = np.stack((self.sample_1, self.sample_2), axis=1)
        sample_A, sample_B = [], []
        # logger.info(f'INITAL: {iterativeCov.getCovariance()}')
        for i,s in enumerate(sample):
            logger.info(f"---- NEW SAMPLE ----")
            sample_A.append(s[0])
            sample_B.append(s[1])
            #compute_curr(sample_A, sample_B)
            self.iterativeCov.increment(data_1= [s[0]], data_2= [s[1]])
            
            if i > 0 :
                yield {'mu_1' : self.iterativeCov.get_mean1()[0], 
                        'mu_2' : self.iterativeCov.get_mean2()[0],
                        'cov' : self.iterativeCov.getCovariance()[0],
                        'gt_mu_1' : np.mean(sample_A),
                        'gt_mu_2' : np.mean(sample_B),
                        'gt_cov' : np.cov(sample_A, sample_B)[0][1]}

class TestIterativeVariance(unittest.TestCase):
      
    def test_1(self):
        sample_1 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        sample_2 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        
        test_cov = TestCov(sample_1, sample_2)
        gener = test_cov.run()
        while True :
            try :
                next_pred = next(gener)
                logger.info(f'next_pred= {next_pred}')
                self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

    # def test_2(self):
    #     sample_1 = np.array([1000.0, 1001.0, 1002.0])
    #     sample_2 = np.array([1000.0, 1001.0, 1002.0])
        
    #     test_cov = TestCov(sample_1, sample_2)
    #     gener = test_cov.run()
    #     while True :
    #         try :
    #             next_pred = next(gener)
    #             self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
    #             self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
    #             self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
    #         except StopIteration:
    #             logger.debug('Stop')
    #             break

    # def test_3(self):
    #     delta = 1.e-10 
    #     sample_1 = np.array([1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta])
    #     sample_2 = np.array([1000.0, 1001.0, 1002.0])
        
    #     test_cov = TestCov(sample_1, sample_2)
    #     gener = test_cov.run()
    #     while True :
    #         try :
    #             next_pred = next(gener)
    #             self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
    #             self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
    #             self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
    #         except StopIteration:
    #             logger.debug('Stop')
    #             break

