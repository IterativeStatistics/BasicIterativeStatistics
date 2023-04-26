import unittest
import numpy as np



from iterative_stats.iterative_covariance import IterativeCovariance
from iterative_stats.utils.logger import logger

class TestCov:
    def __init__(self, sample_1, sample_2, dim: int = 1):
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.nb_sample = len(sample_1)
        self.dim = dim
        self.iterativeCov = IterativeCovariance(dim=dim)
        
    
    def run(self):
        for i in range(self.nb_sample):
            # logger.info(f"---- NEW SAMPLE {i} ----")
            self.iterativeCov.increment(data_1= self.sample_1[i], data_2= self.sample_2[i])

            if i > 0 :
                if self.dim == 1 : 
                    cov =  np.cov(self.sample_1[:(i+1)], self.sample_2[:(i+1)])[0][0]
                else :
                    cov = np.zeros(self.dim)
                    for k in range(self.dim):
                        cov[k] = np.cov(self.sample_1[:(i+1), k], self.sample_2[:(i+1), k])[0][1]
                yield {'mu_1' : self.iterativeCov.get_mean1(), 
                        'mu_2' : self.iterativeCov.get_mean2(),
                        'cov' : self.iterativeCov.getCovariance(),
                        'gt_mu_1' : np.mean(self.sample_1[:(i+1)], axis=0),
                        'gt_mu_2' : np.mean(self.sample_2[:(i+1)], axis=0),
                        'gt_cov' : cov}

class TestIterativeVariance(unittest.TestCase):
      
    def test_1(self):
        sample_1 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        sample_2 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        
        test_cov = TestCov(sample_1, sample_2)
        gener = test_cov.run()
        while True :
            try :
                next_pred = next(gener)
                # logger.info(f'next_pred= {next_pred}')
                self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

    def test_2(self):
        sample_1 = np.array([1000.0, 1001.0, 1002.0])
        sample_2 = np.array([1000.0, 1001.0, 1002.0])
        
        test_cov = TestCov(sample_1, sample_2)
        gener = test_cov.run()
        while True :
            try :
                next_pred = next(gener)
                self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

    def test_3(self):
        delta = 1.e-10 
        sample_1 = np.array([1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta])
        sample_2 = np.array([1000.0, 1001.0, 1002.0])
        
        test_cov = TestCov(sample_1, sample_2)
        gener = test_cov.run()
        while True :
            try :
                next_pred = next(gener)
                self.assertAlmostEqual(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), delta=10e-10)
                self.assertAlmostEqual(next_pred.get('gt_cov'), next_pred.get('cov'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

    def test_4_multidim(self):
        delta = 1.e-10 
        sample_1 = np.array([[10.0, 11.0, 12.0], [1000.0, 1001.0, 1002.0]])
        sample_2 = np.array([[10.0, 11.0, 12.0], [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta]])
        
        test_cov = TestCov(np.transpose(sample_1), np.transpose(sample_2), dim=2)
        gener = test_cov.run()
        while True :
            try :
                next_pred = next(gener)
                logger.info(f'next_pred: {next_pred}')
                # Check mu1
                res = np.allclose(next_pred.get('gt_mu_1'), next_pred.get('mu_1'), atol=10e-10)
                self.assertTrue(res)
                # Check mu2
                res = np.allclose(next_pred.get('gt_mu_2'), next_pred.get('mu_2'), atol=10e-10)
                self.assertTrue(res)
                # Check cov
                res = np.allclose(next_pred.get('gt_cov'), next_pred.get('cov'), atol=10e-10)
                self.assertTrue(res)
            except StopIteration:
                logger.debug('Stop')
                break