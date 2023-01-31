import unittest
import copy 

import numpy as np
import openturns as ot
from iterative_stats.sobol.sobol_jansen import IterativeJansenSobol

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestIterativeSobolJansen(unittest.TestCase):

    def test_ishigami_ot(self):
        # Draw samples A and B (here A = (X1, X2, X3), where X1, X2 et X3 are iid and follows a Unif[a,b])
        nb_parms = 3
        sobol = IterativeJansenSobol({'vector_size' :1, 'nb_parms':nb_parms})
        nb_sim = 3

        # Create the model and input distribution
        formula = ['sin(pi_*X1)+7*sin(pi_*X2)^2+0.1*(pi_*X3)^4*sin(pi_*X1)']
        model = ot.SymbolicFunction(['X1', 'X2', 'X3'], formula)
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3)
        ot.RandomGenerator.SetSeed(0)
        inputDesign = ot.SobolIndicesExperiment(distribution, nb_sim).generate()
        outputDesign = model(inputDesign)


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
            logger.info(f'Sobol iterative: first = {sobol.getFirstOrderSobol()}, tot = {sobol.getTotalOrderSobol()}')
           
       
        first_order = sobol.getFirstOrderSobol()
        total_order = sobol.getTotalOrderSobol()
        
        sensitivityAnalysis = ot.JansenSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        ot_first_order = sensitivityAnalysis.getFirstOrderIndices()
        ot_total_order = sensitivityAnalysis.getTotalOrderIndices()
        logger.info(f'Sobol openturns: first = {ot_first_order}, total order {ot_total_order}')
        for p in range(nb_parms):
            self.assertAlmostEqual(ot_first_order[p], first_order[p], delta=10e-10)
            self.assertAlmostEqual(total_order[p], ot_total_order[p], delta=10e-10)
