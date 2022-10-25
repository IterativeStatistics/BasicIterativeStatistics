import unittest
import numpy as np

# Initialize the logger
import logging
logger = logging.getLogger("my-logger")

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics



class TestAbstractIterativeStatistics(unittest.TestCase):
    class MockAbstractIterativeStatistics(AbstractIterativeStatistics):
        def increment(self, data : np.array):
            self.state += data
      
    def test_init(self):
        try :
            mock = self.MockAbstractIterativeStatistics({'vector_size': 10})
            self.assertTrue(True)
            np.testing.assert_equal(mock.get_stats(), np.zeros(10))
        except Exception as e :
            logger.info(f'Raise exception: {e}')
            self.assertTrue(False)

    def test_increment(self):
        mock = self.MockAbstractIterativeStatistics({'vector_size': 10})
        state = mock.get_stats()

        data = np.ones(10)
        mock.increment(data)
        self.assertListEqual(list(np.ones(10)), list(mock.get_stats()))


