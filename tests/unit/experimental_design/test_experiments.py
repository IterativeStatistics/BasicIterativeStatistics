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

    def test_instance_1(self):
        gen = Uniform3D(nb_parms = self.nb_parms, nb_sim = self.nb_sim, 
                        l_bounds = self.l_bounds, u_bounds = self.u_bounds, second_order=False).generator()
        
        while True :
            try :
                sample = next(gen)
                for i in range(self.nb_parms):
                    self.assertTrue((sample[:,i] <=self.u_bounds[i]).any())
                    self.assertTrue((sample[:,i] >= self.l_bounds[i]).any())
                self.assertEqual(len(sample), (2 +  self.nb_parms))
            except StopIteration:
                break 


    def test_instance_2(self):
        gen = Uniform3D(nb_parms = self.nb_parms, nb_sim = self.nb_sim, 
                        l_bounds = self.l_bounds, u_bounds = self.u_bounds, second_order=True).generator()
        
        while True :
            try :
                sample = next(gen)
                for i in range(self.nb_parms):
                    self.assertTrue((sample[:,i] <=self.u_bounds[i]).any())
                    self.assertTrue((sample[:,i] >= self.l_bounds[i]).any())
                self.assertEqual(len(sample), (2 + 2 * self.nb_parms))
            except StopIteration:
                break 

        