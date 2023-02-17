import unittest
import copy 

import numpy as np
import openturns as ot
from iterative_stats.sobol.sobol_jansen import IterativeJansenSobol

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CheckSobolIndices:
    def __init__(self, nb_parms, nb_sim = 1):
        self.data_A = np.array([])
        self.data_B = np.array([])
        self.data_E = np.array([]*nb_parms)
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.iteration = 0

    def _compute_dotproduct(self, data_1, data_2):
        vec = []
        for p in range(self.nb_parms):
            vec.append(np.dot(data_2[:,p]-data_1, data_2[:,p]-data_1))
        return np.array(vec)

    def _compute_centeredsquare(self, data):
        vec = []
        for p in range(self.nb_parms):
            vec.append(np.dot(data, data))
        return np.array(vec)

    def increment(self, sample):
        self.iteration += 1
        if self.data_B.size == 0 and self.data_E.size == 0 :
            self.data_A = sample[:self.nb_sim]
            self.data_B = sample[self.nb_sim:2*self.nb_sim]
            self.data_E = sample[2*self.nb_sim:]
        else :
            self.data_A = np.append(self.data_A, sample[:self.nb_sim])
            self.data_B = np.append(self.data_B, sample[self.nb_sim:2*self.nb_sim])
            self.data_E = np.vstack((self.data_E, sample[2*self.nb_sim:]))

        if self.iteration > 1 :
            var = np.var(self.data_A, ddof = 1)
            vi_first_order = var - self._compute_dotproduct(self.data_B, self.data_E)/(2*self.iteration - 1)

            vi_total_order = self._compute_dotproduct(self.data_A, self.data_E)/(2*self.iteration - 1)
            return {'first_order' : vi_first_order/var,
                    'total_order' : vi_total_order/var
                    }
        else :
            return None 

class TestIterativeSobolJansen(unittest.TestCase):

    def test_ishigami_ot(self):
        # Draw samples A and B (here A = (X1, X2, X3), where X1, X2 et X3 are iid and follows a Unif[a,b])
        nb_parms = 3
        sobol = IterativeJansenSobol({'vector_size' :1, 'nb_parms':nb_parms})
        nb_sim = 20

        # Create the model and input distribution
        formula = ['sin(pi_*X1)+7*sin(pi_*X2)^2+0.1*(pi_*X3)^4*sin(pi_*X1)']
        model = ot.SymbolicFunction(['X1', 'X2', 'X3'], formula)
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3)
        ot.RandomGenerator.SetSeed(0)
        inputDesign = ot.SobolIndicesExperiment(distribution, nb_sim).generate()
        outputDesign = model(inputDesign)

        check = CheckSobolIndices(nb_parms = nb_parms)

        # Check the iterative algorithm
        # -- Apply the pick-freeze approach
        for i in range(nb_sim):
            sample_A = inputDesign[i]
            sample_B = inputDesign[nb_sim + i]
            sample = np.vstack(([outputDesign[i]], [outputDesign[nb_sim + i]]))
            for k in range(nb_parms):
                sample_Ck = copy.deepcopy(sample_A)  
                sample_Ck[k] = sample_B[k]
                sample = np.append(sample, model(sample_Ck)[0])
            sobol.increment(sample)
            pearson = check.increment(sample)
            
       
        first_order = sobol.getFirstOrderSobol()
        total_order = sobol.getTotalOrderSobol()

        if pearson is not None :
            for p in range(nb_parms):
                # check first order
                self.assertAlmostEqual(pearson.get('first_order')[p], first_order[p], delta=10e-10)
                # check total order
                self.assertAlmostEqual(pearson.get('total_order')[p], total_order[p], delta=10e-10)
            

        sensitivityAnalysis = ot.JansenSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        ot_first_order = sensitivityAnalysis.getFirstOrderIndices()
        ot_total_order = sensitivityAnalysis.getTotalOrderIndices()
        
        for p in range(nb_parms):
            self.assertAlmostEqual(ot_first_order[p], first_order[p], delta=10e-10)
            self.assertAlmostEqual(total_order[p], ot_total_order[p], delta=10e-10)
