import unittest
import copy 

import numpy as np
import openturns as ot
from iterative_stats.sobol.sobol_martinez import IterativeMartinezSobol

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class CheckSobolIndices:
    def __init__(self, nb_parms, nb_sim = 1):
        self.data_A = np.array([])
        self.data_B = np.array([])
        self.data_E = np.array([]*nb_parms)
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim

    def _compute_pearson(self, data_1, data_2):
        pearson = np.empty(self.nb_parms)
        for p in range(self.nb_parms) :
            cov = np.cov(data_1, data_2[:,p])[0][1]
            pearson[p] = cov / (np.std(data_1, ddof=1)*np.std(data_2[:,p], ddof=1))
        return pearson

    def increment(self, sample):
        if self.data_B.size == 0 and self.data_E.size == 0 :
            self.data_A = sample[:self.nb_sim]
            self.data_B = sample[self.nb_sim:2*self.nb_sim]
            self.data_E = sample[2*self.nb_sim:]
        else :
            self.data_A = np.append(self.data_A, sample[:self.nb_sim])
            self.data_B = np.append(self.data_B, sample[self.nb_sim:2*self.nb_sim])
            self.data_E = np.vstack((self.data_E, sample[2*self.nb_sim:]))

        if self.data_B.size > 1 :
            return {'first_order' : self._compute_pearson(self.data_B, self.data_E),
                    'total_order' : 1 - self._compute_pearson(self.data_A, self.data_E)
                    }
        else :
            return None 


class TestIterativeSobolMartinez(unittest.TestCase):

    def test_ishigami_ot(self):
        # Draw samples A and B (here A = (X1, X2, X3), where X1, X2 et X3 are iid and follows a Unif[a,b])
        nb_parms = 3
        sobol = IterativeMartinezSobol({'vector_size' :1, 'nb_parms':nb_parms})
        nb_sim = 10

        # Create the model and input distribution
        formula = ['sin(pi_*X1)+7*sin(pi_*X2)^2+0.1*(pi_*X3)^4*sin(pi_*X1)']
        model = ot.SymbolicFunction(['X1', 'X2', 'X3'], formula)
        distribution = ot.ComposedDistribution([ot.Uniform(-1.0, 1.0)] * 3)
        ot.RandomGenerator.SetSeed(0)
        inputDesign = ot.SobolIndicesExperiment(distribution, nb_sim).generate()
        outputDesign = model(inputDesign)

        check_pearson = CheckSobolIndices(nb_parms = nb_parms)

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

            logger.info(f'sample: {sample}')
            sobol.increment(sample)
            pearson = check_pearson.increment(sample)
            logger.info(f'Sobol iterative: first = {sobol.getFirstOrderSobol()}, tot = {sobol.getTotalOrderSobol()}')
            logger.info(f'Sobol pearson : {pearson}')

            if pearson is not None :
                first_order = sobol.getFirstOrderSobol()
                total_order = sobol.getTotalOrderSobol()
                for p in range(nb_parms):
                    # check first order
                    self.assertAlmostEqual(pearson.get('first_order')[p], first_order[p], delta=10e-10)
                    # check total order
                    self.assertAlmostEqual(pearson.get('total_order')[p], total_order[p], delta=10e-10)
                   
       
        sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        ot_first_order = sensitivityAnalysis.getFirstOrderIndices()
        ot_total_order = sensitivityAnalysis.getTotalOrderIndices()
        logger.info(f'Sobol openturns: first = {ot_first_order}, total order {ot_total_order}')
        for p in range(nb_parms):
            self.assertAlmostEqual(ot_first_order[p], first_order[p], delta=10e-10)
            self.assertAlmostEqual(total_order[p], ot_total_order[p], delta=10e-10)
