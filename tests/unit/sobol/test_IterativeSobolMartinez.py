from random import sample
import unittest
import copy 

import numpy as np
import openturns as ot
from src.sobol.sobol_martinez import IterativeSobol

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# TODO: 
#   - Tester avec les fonctions ishigami
#   - Tester avec OpenTurns
#
#            
       
class TestIterativeCovariance(unittest.TestCase):


    # @classmethod
    # def setUp(self):



    def test_ishigami_ot(self):
        # Draw samples A and B (here A = (X1, X2, X3), where X1, X2 et X3 are iid and follows a Unif[a,b])
        nb_parms = 3
        sobol = IterativeSobol({'vector_size' :1, 'nb_parms':nb_parms})
        nb_sim = 10000

        # Create the model and input distribution
        a, b, c = 1, 7 , 0.1
        formula = [f'{a}*sin(pi_*X1)+{b}*sin(X2)*sin(pi_*X2)+' + \
           f'{c}*((pi_*X3)^4)*sin(pi_*X1)']
        input_names = ['X1', 'X2', 'X3']
        model = ot.SymbolicFunction(input_names, formula)
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3, \
                                        ot.IndependentCopula(3))
        dimension = distribution.getDimension()
        ot.RandomGenerator.SetSeed(0)
        inputDesign = ot.SobolIndicesExperiment(distribution, 2*nb_sim, True).generate()
        outputDesign = model(inputDesign)

        # with numpy
        numpy_sample_C = [np.zeros(nb_sim) for _ in range(nb_parms)]
        numpy_sample_B = np.zeros(nb_sim)
        # Apply the pick-freeze approach
        for i in range(nb_sim):
            sample_A = inputDesign[2*i]
            sample_B = inputDesign[2*i + 1]
            
            y_C = []
            for k in range(nb_parms):
                sample_Ck = copy.deepcopy(sample_A)  
                sample_Ck[k] = sample_B[k]
                y_C.append(model(sample_Ck)[0])
                
                numpy_sample_C[k][i] = model(sample_Ck)[0]
            
            numpy_sample_B[i] = model(sample_B)[0]
                
            sobol.increment(np.array(outputDesign[2*i+1]), y_C)
        
        logger.info(f'Sobol iterative: {sobol.get_stats()}')


        logger.info(f'Sobol numpy ------------')
        #logger.info(f'Sobol numpy_sample_C: sample={numpy_sample_C[0]}')
        # logger.info(f'Sobol numpy_sample_B: sample={numpy_sample_B}')
        logger.info(f'Sobol type: {type(numpy_sample_B)} -- {type(numpy_sample_C[0])}')
        np_sobol = np.zeros(nb_parms)
        for k in range(nb_parms):
            prod_var = np.var(numpy_sample_B,ddof=1)*np.var(numpy_sample_C[k],ddof=1)
            np_sobol[k] = np.cov(numpy_sample_B, numpy_sample_C[k])[0][1]/np.sqrt(prod_var)
        logger.info(f'Sobol numpy: {np_sobol}')

        # Compute first order indices using the Saltelli estimator
        sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, 2*nb_sim)
        first_order = sensitivityAnalysis.getFirstOrderIndices()
        logger.info(f'Sobol openturns: {first_order}')