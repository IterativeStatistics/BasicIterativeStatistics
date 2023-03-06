import unittest
import numpy as np 

from iterative_stats.sensitivity.sensitivity_martinez import IterativeSensitivityMartinez

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()



class TestIterativeSensitivityMartinez(unittest.TestCase):

    def test_ishigami(self):
        from tests.mock.sensitivity.check_ishigami import check_ishigami
        nb_parms = 3
        nb_sim = 20
        sensitivity_indices = IterativeSensitivityMartinez(vector_size = 1, nb_parms = nb_parms, second_order = True)

        from tests.mock.sensitivity.check_martinez import MartinezCheckSensitivityIndices
        check_sensitivity = MartinezCheckSensitivityIndices(nb_parms = nb_parms, second_order = True)

        check_ishigami(nb_parms, nb_sim, sensitivity_indices, check_sensitivity, second_order = True)

        check_firstorderindices = check_sensitivity.compute_firstorderindices()
        check_secondorderindices = check_sensitivity.compute_secondorderindices()
        check_totalorderindices = check_sensitivity.compute_totalorderindices()

        iterative_firstorderindices = sensitivity_indices.getFirstOrderIndices()
        iterative_secondorderindices = sensitivity_indices.getSecondOrderIndices()
        iterative_secondorderindices_correct = [[iterative_secondorderindices[i][j][0] for i in range(nb_parms)] for j in range(nb_parms)]
        iterative_secondorderindices = iterative_secondorderindices_correct
        iterative_totalorderindices = sensitivity_indices.getTotalOrderIndices()

        for p in range(nb_parms):
            # check first order
            self.assertTrue(np.allclose(check_firstorderindices[p], iterative_firstorderindices[p], atol=10e-10))
            # check second order
            self.assertTrue(np.allclose(check_secondorderindices[p], iterative_secondorderindices[p], atol=10e-10))
            # check total order
            self.assertTrue(np.allclose(check_totalorderindices[p], iterative_totalorderindices[p], atol=10e-10))

    def test_ishigami_with_openturns(self):
        import openturns as ot

        nb_parms = 3
        nb_sim = 20
        sensitivity_indices = IterativeSensitivityMartinez(vector_size = 1, nb_parms =nb_parms)

        from tests.mock.sensitivity.check_ishigami import ishigami_with_openturns
        inputDesign, outputDesign = ishigami_with_openturns(nb_parms, nb_sim, sensitivity_indices)

        iterative_firstorderindices = sensitivity_indices.getFirstOrderIndices()
        # iterative_secondorderindices = sensitivity_indices.getSecondOrderIndices()
        iterative_totalorderindices = sensitivity_indices.getTotalOrderIndices()

        sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        ot_first_order = sensitivityAnalysis.getFirstOrderIndices()
        # ot_second_order = sensitivityAnalysis.getSecondOrderIndices()
        ot_total_order = sensitivityAnalysis.getTotalOrderIndices()
        
        for p in range(nb_parms):
            self.assertAlmostEqual(ot_first_order[p], iterative_firstorderindices[p], delta=10e-10)
            # self.assertAlmostEqual(ot_second_order[p], iterative_secondorderindices[p], delta=10e-10)
            self.assertAlmostEqual(ot_total_order[p], iterative_totalorderindices[p], delta=10e-10)


