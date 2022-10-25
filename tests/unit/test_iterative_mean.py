import unittest
import numpy as np

# Initialize the logger
import logging
logger = logging.getLogger("my-logger")

from src.iterative_mean import IterativeMean



class TestIterativeMean(unittest.TestCase):

    def test_init(self):
        try :
            iter_mean = IterativeMean({'dimension': 10})
            self.assertTrue(True)
            self.assertIsNone(iter_mean.get_stats())
        except Exception as e :
            logger.info(f'Raise exception: {e}')
            self.assertTrue(False)

    def test_update(self):
        iter_mean = IterativeMean({'dimension': 10})

        data = np.ones(10)
        iter_mean.update(data)
        iter_mean.update(data)
        self.assertListEqual(list(np.ones(10)), list(iter_mean.get_stats()))

