import unittest
import numpy as np



from iterative_stats.iterative_dotproduct import IterativeDotProduct
from iterative_stats.utils.logger import logger

class TestIterative:
    def __init__(self, sample_1, sample_2, shift):
        self.sample_1 = sample_1
        self.sample_2 = sample_2
        self.nb_sample = len(sample_1)
        self.shift = shift
        self.iterative_value = IterativeDotProduct()
    
    def run(self):
        for i in range(self.nb_sample):
            logger.info(f"---- NEW SAMPLE {i} ----")
            self.iterative_value.increment(data_1= self.sample_1[i], 
                                            data_2= self.sample_2[i],
                                            shift = self.shift[i])
            
            if i > 0 :
                logger.info(f'i= {i}')
                logger.info(f'sample: {self.sample_1[:i] - self.shift[i]}')
                logger.info(f'sample 1: {self.sample_1[:i]}')
                logger.info(f'shift: {self.shift[i]}')
                gt = np.dot(self.sample_1[:(i+1)] - self.shift[i], 
                            self.sample_2[:(i+1)] - self.shift[i])
                yield {'iterative_val' : self.iterative_value.get_stats(), 
                        'gt_iterative_val' : gt / i}

class TestIterativeDotProduct(unittest.TestCase):
      
    def test_1(self):
        sample_1 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        sample_2 = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        shift = np.array([-10.0, 5.0, -12.0, 13.0, -11.0, -12.2])
        
        test_iterative = TestIterative(sample_1, sample_2, shift)
        gener = test_iterative.run()
        while True :
            try :
                next_pred = next(gener)
                logger.info(f'next_pred= {next_pred}')
                self.assertAlmostEqual(next_pred.get('iterative_val'), next_pred.get('gt_iterative_val'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

    def test_2(self):
        sample_1 = np.array([1000.0, 1001.0, 1002.0])
        sample_2 = np.array([1000.0, 1001.0, 1002.0])
        shift = np.array([-10.0, 5.0, -12.0, 13.0, -11.0, -12.2])
        
        test_iterative = TestIterative(sample_1, sample_2, shift)
        gener = test_iterative.run()
        while True :
            try :
                next_pred = next(gener)
                logger.info(f'next_pred= {next_pred}')
                self.assertAlmostEqual(next_pred.get('iterative_val'), next_pred.get('gt_iterative_val'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break


    def test_3(self):
        delta = 1.e-10 
        sample_1 = np.array([1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta])
        sample_2 = np.array([1000.0, 1001.0, 1002.0])
        shift = np.array([-10.0, 5.0, -12.0, 13.0, -11.0, -12.2])
        
        test_iterative = TestIterative(sample_1, sample_2, shift)
        gener = test_iterative.run()
        while True :
            try :
                next_pred = next(gener)
                logger.info(f'next_pred= {next_pred}')
                self.assertAlmostEqual(next_pred.get('iterative_val'), next_pred.get('gt_iterative_val'), delta=10e-10)
            except StopIteration:
                logger.debug('Stop')
                break

