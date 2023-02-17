import unittest
import copy 

import numpy as np
import openturns as ot
from tests.mock.uniform_3d import Uniform3D

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class TestExperimentalDesign(unittest.TestCase):

    def setUp(self):
        self.nb_parms = 3
        self.nb_sim = 10
        self.l_bounds = [-2, -2, -2]
        self.u_bounds = [2, 2, 2]

    def test_instance(self):
        exp = Uniform3D(nb_parms = self.nb_parms, nb_sim = self.nb_sim, 
                        l_bounds = self.l_bounds, u_bounds = self.u_bounds)
        
        for i in range(self.nb_sim) :
            sample = exp.draw()[0]
            # logger.info(f'sample {i} = {sample}')
            for i in range(self.nb_parms):
                self.assertTrue(sample[i] <=self.u_bounds[i] and sample[i] >= self.l_bounds[i])