import unittest
import numpy as np

# Initialize the logger
import logging
logger = logging.getLogger("my-logger")

from src.iterative_moment import IterativeMoment



class TestIterativeMoment(unittest.TestCase):

    def test_init(self):
        try :
            iter_mean = IterativeMean({'dimension': 10, 'moments' : []})
            self.assertTrue(True)
            self.assertIsNone(iter_mean.get_stats())
        except Exception as e :
            logger.info(f'Raise exception: {e}')
            self.assertTrue(False)

    def test_update(self):
        iter_moment = IterativeMoment({'dimension': 10, 'moments' : [1]})

        data = np.ones(10)
        iter_moment.update(data)
        iter_moment.update(data)
        self.assertListEqual(list(np.ones(10)), list(iter_moment.get_mean()))
        self.assertIsNone(iter_moment.get_variance())

