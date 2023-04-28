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
            mock = self.MockAbstractIterativeStatistics(dim= 10)
            self.assertTrue(True)
            np.testing.assert_equal(mock.get_stats(), np.zeros(10))
        except Exception as e :
            logger.info(f'Raise exception: {e}')
            self.assertTrue(False)

    def test_increment(self):
        mock = self.MockAbstractIterativeStatistics(dim= 10)

        data = np.ones(10)
        mock.increment(data)
        self.assertListEqual(list(np.ones(10)), list(mock.get_stats()))


    def test_save_state(self):
        mock = self.MockAbstractIterativeStatistics(dim= 10)

        data = np.ones(10)
        mock.increment(data)
        state = mock.save_state()
        mock_ft = self.MockAbstractIterativeStatistics(dim= 10, state=state)
        mock_ft.increment(data)
        self.assertListEqual(list(np.ones(10)*2.), list(mock_ft.get_stats()))

