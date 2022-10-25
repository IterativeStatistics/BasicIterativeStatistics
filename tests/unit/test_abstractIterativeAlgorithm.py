import unittest
import numpy as np

# Initialize the logger
import logging
logger = logging.getLogger("my-logger")

from src.abstract_iterative_algorithm import AbstractIterativeAlgorithm



class TestAbstractIterativeAlgorithm(unittest.TestCase):
    class MockAbstractIterativeAlgorithm(AbstractIterativeAlgorithm):
        def update(self, data : np.array):
            if self.iteration == 0 :
                self.state = data
            else :
                self.state += data
      
    def test_init(self):
        try :
            mock = self.MockAbstractIterativeAlgorithm({'dimension': 10})
            self.assertTrue(True)
            self.assertIsNone(mock.get_stats())
        except Exception as e :
            logger.info(f'Raise exception: {e}')
            self.assertTrue(False)

    def test_update(self):
        mock = self.MockAbstractIterativeAlgorithm({'dimension': 10})
        state = mock.get_stats()

        data = np.ones(10)
        mock.update(data)
        self.assertListEqual(list(np.ones(10)), list(mock.get_stats()))


