import unittest
import numpy as np

from iterative_stats.iterative_mean import IterativeMean
from iterative_stats.iterative_mean import IterativeShiftedMean

from iterative_stats.utils.logger import logger 

class TestMeanIterativeStatistics(unittest.TestCase):
      
    def test_increment_iterativeMean(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        mu = np.mean(sample)
        iterativeMean = IterativeMean(dim = 1)
        for x in sample :
            iterativeMean.increment(x)
        self.assertAlmostEqual(mu, iterativeMean.get_stats()[0], delta=10e-2)
        self.assertTupleEqual(iterativeMean.get_stats().shape, (1,))

    def test_increment_faulttolerance(self):
        sample = np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        mu = np.mean(sample)
        iterativeMean = IterativeMean(dim = 1)
        for x in sample[:3]:
            iterativeMean.increment(x)
        
        state = iterativeMean.save_state()
        iterativeMean_reload = IterativeMean(dim=1, state=state)
        for x in sample[3:]:
            iterativeMean_reload.increment(x)

        self.assertAlmostEqual(mu, iterativeMean_reload.get_stats(), delta=10e-2)
        self.assertTupleEqual(iterativeMean_reload.get_stats().shape, (1,))

    def test_increment_iterativeMean_multidim(self):
        delta = 1.e-10 
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)
        mu = np.mean(sample, axis=0)
        iterativeMean = IterativeMean(dim = 2)
        for x in sample :
            iterativeMean.increment(x)
        res = np.allclose(mu,iterativeMean.get_stats(), atol=10e-10)
        self.assertTrue(res)
        self.assertTupleEqual(iterativeMean.get_stats().shape, (2,))

    def test_increment_multidim_fault_tolerance(self):
        delta = 1.e-10 
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)
        mu = np.mean(sample, axis=0)
        iterativeMean = IterativeMean(dim = 2)

        for x in sample[:3] :
            iterativeMean.increment(x)
        
        state = iterativeMean.save_state()
        iterativeMean_reload = IterativeMean(dim=2, state=state)

        for x in sample[3:] :
            iterativeMean_reload.increment(x)

        self.assertTrue(np.allclose(mu,iterativeMean_reload.get_stats(), atol=10e-10))
        self.assertTupleEqual(iterativeMean_reload.get_stats().shape, (2,))

class TestMeanShiftedIterativeStatistics(unittest.TestCase):

    def test_increment_iterativeShiftedMean(self):
        sample =  np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        shifted_sample = np.array([-1., 5., -6., 13.0, -16.0, -12.2])
        iterativeMean = IterativeShiftedMean(dim = 1)

        cpt = 0
        for x, shift in zip(sample, shifted_sample) :
            iterativeMean.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt])
            logger.info(f'mu: {mu}, other={iterativeMean.get_stats()}')
            cpt += 1
            self.assertAlmostEqual(mu, iterativeMean.get_stats(), delta=10e-2)
            self.assertTupleEqual(iterativeMean.get_stats().shape, (1,))

    def test_increment_iterativeShiftedMean_fault_tolerance(self):
        sample =  np.array([10.0, 11.0, 12.0, 13.0, 16.0, -12.2])
        shifted_sample = np.array([-1., 5., -6., 13.0, -16.0, -12.2])
        iterativeMean = IterativeShiftedMean(dim = 1)

        cpt = 0
        for x, shift in zip(sample[:3], shifted_sample[:3]) :
            iterativeMean.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt])
            logger.info(f'mu: {mu}, other={iterativeMean.get_stats()}')
            cpt += 1
            self.assertAlmostEqual(mu, iterativeMean.get_stats(), delta=10e-2)
            self.assertTupleEqual(iterativeMean.get_stats().shape, (1,))
        
        state = iterativeMean.save_state()
        iterativeMean_reload = IterativeShiftedMean(dim=1, state=state)

        for x, shift in zip(sample[3:], shifted_sample[3:]) :
            iterativeMean_reload.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt])
            logger.info(f'(reload) mu: {mu}, other={iterativeMean_reload.get_stats()}')
            cpt += 1
            self.assertAlmostEqual(mu, iterativeMean_reload.get_stats(), delta=10e-2)
            self.assertTupleEqual(iterativeMean_reload.get_stats().shape, (1,))

    def test_increment_iterativeShiftedMean_multidim(self):
        delta = 1.e-10 
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], 
                            [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)
        shifted_sample = np.array([[-1., 5., -6., 13.0, -16.0, -12.2], 
                                    [-0.5, 4., -1., 3.0, -6.0, -2.2]])
        shifted_sample = np.transpose(shifted_sample)
        iterativeMean = IterativeShiftedMean(dim = 2)

        cpt = 0
        for x, shift in zip(sample, shifted_sample) :
            iterativeMean.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt], axis=0)
            # logger.info(f'mu: {mu}, other={iterativeMean.get_stats()}')
            cpt += 1
            res = np.allclose(mu,iterativeMean.get_stats(), atol=10e-10)
            self.assertTrue(res)
            self.assertTupleEqual(iterativeMean.get_stats().shape, (2,))

    def test_increment_iterativeShiftedMean_multidim_faulttolerance(self):
        delta = 1.e-10 
        sample = np.array([[10.0, 11.0, 12.0, 13.0, 16.0, -12.2], 
                            [1.0 + delta, 1.0 + 2.0 * delta, 1.0 + 3 * delta, 13.0, 16.0, -12.2]])
        sample = np.transpose(sample)
        shifted_sample = np.array([[-1., 5., -6., 13.0, -16.0, -12.2], 
                                    [-0.5, 4., -1., 3.0, -6.0, -2.2]])
        shifted_sample = np.transpose(shifted_sample)
        iterativeMean = IterativeShiftedMean(dim = 2)

        cpt = 0
        for x, shift in zip(sample[:3], shifted_sample[:3]) :
            iterativeMean.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt], axis=0)
            # logger.info(f'mu: {mu}, other={iterativeMean.get_stats()}')
            cpt += 1
            res = np.allclose(mu,iterativeMean.get_stats(), atol=10e-10)
            self.assertTrue(res)
            self.assertTupleEqual(iterativeMean.get_stats().shape, (2,))

        state = iterativeMean.save_state()
        iterativeMean_reload = IterativeShiftedMean(dim=1, state=state)

        for x, shift in zip(sample[3:], shifted_sample[3:]) :
            iterativeMean_reload.increment(x, shift = shift)
            mu = np.mean(sample[:cpt+1] - shifted_sample[cpt], axis=0)
            logger.info(f'(reload) mu: {mu}, other={iterativeMean_reload.get_stats()}')
            cpt += 1
            res = np.allclose(mu,iterativeMean.get_stats(), atol=10e-10)
            self.assertTrue(res)
            self.assertTupleEqual(iterativeMean.get_stats().shape, (2,))