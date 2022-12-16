from random import sample
import unittest
import copy 

import numpy as np
import openturns as ot
from iterative_stats.sobol.sobol_martinez import IterativeSobol
from experimental_design.experiment import AbstractExperiment

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# TODO: 
#   - Tester avec les fonctions ishigami
#   - Tester avec OpenTurns

class PickFreeze(AbstractExperiment):
    def __init__(self, conf) -> None:  
        nb_parms = conf.get('nb_parms')
        self.numpy_sample_C = [np.zeros(self.sample_size) for _ in range(nb_parms)]
        self.numpy_sample_B = np.zeros(self.sample_size)  
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3) 
        self.inputDesign = ot.SobolIndicesExperiment(distribution, self.sample_size).generate() 
        formula = [conf.get('formula')]
        self.model = ot.SymbolicFunction(['X1', 'X2', 'X3'], formula)
        self.cpt = 0

    def draw(self) :
        
        # Apply the pick-freeze approach

        sample_A = self.inputDesign[2*self.cpt]
        sample_B = self.inputDesign[2*self.cpt + 1]
        y_C = []
        for k in range(self.nb_parms):
            sample_Ck = copy.deepcopy(sample_A)  
            sample_Ck[k] = sample_B[k]
            y_C.append(self.model(sample_Ck)[0])
            self.numpy_sample_C[k][self.cpt] = self.model(sample_Ck)[0]
        self.numpy_sample_B[self.cpt] = self.model(sample_B)[0]
        return 

class TestIterativeCovariance(unittest.TestCase):


    # @classmethod
    # def setUp(self):



    def test_ishigami_ot(self):
        # Draw samples A and B (here A = (X1, X2, X3), where X1, X2 et X3 are iid and follows a Unif[a,b])
        nb_parms = 3
        sobol = IterativeSobol({'vector_size' :1, 'nb_parms':nb_parms})
        nb_sim = 3

        # Create the model and input distribution
        formula = ['sin(pi_*X1)+7*sin(pi_*X2)^2+0.1*(pi_*X3)^4*sin(pi_*X1)']
        model = ot.SymbolicFunction(['X1', 'X2', 'X3'], formula)
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3)
        dimension = distribution.getDimension()
        ot.RandomGenerator.SetSeed(0)
        inputDesign = ot.SobolIndicesExperiment(distribution, nb_sim).generate()
        outputDesign = model(inputDesign)
        logger.info(f'input size : {inputDesign}')

        # Check the iterative algorithm
        # -- Apply the pick-freeze approach
        numpy_sample_C = [np.zeros(nb_sim) for _ in range(nb_parms)]
        numpy_sample_B = np.zeros(nb_sim)
        for i in range(nb_sim):
            sample_A = inputDesign[i]
            sample_B = inputDesign[nb_sim + i]
            
            y_C = []
            for k in range(nb_parms):
                sample_Ck = copy.deepcopy(sample_A)  
                sample_Ck[k] = sample_B[k]
                logger.info(f'sample CK: {sample_Ck}')
                y_C.append(model(sample_Ck)[0])
            
            sobol.increment(np.array(outputDesign[i]), y_C)
        logger.info(f'Sobol iterative: {sobol.get_stats()}')

        # logger.info(f'Sobol numpy ------------')
        # np_sobol = np.zeros(nb_parms)
        # for k in range(nb_parms):
        #     prod_var = np.var(numpy_sample_B,ddof=1)*np.var(numpy_sample_C[k],ddof=1)
        #     np_sobol[k] = np.cov(numpy_sample_B, numpy_sample_C[k])[0][1]/np.sqrt(prod_var)
        # logger.info(f'Sobol numpy: {np_sobol}')

        # Compute first order indices using the Martinez estimator
        sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        first_order = sensitivityAnalysis.getFirstOrderIndices()
        logger.info(f'Sobol openturns: first order {first_order}')
        total_order = sensitivityAnalysis.getTotalOrderIndices()
        logger.info(f'Sobol openturns: total order {total_order}')
