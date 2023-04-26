

import numpy as np
from iterative_stats.utils.logger import logger
from tests.mock.sensitivity.check_ishigami import check_ishigami, check_ishigami_multi_dim
from iterative_stats.sensitivity import SALTELLI, JANSEN, MARTINEZ


class SensitivityTester_Ishigami:
    def __init__(self, nb_parms, nb_sim, second_order, dim, sensitivity_instance,check_instance):
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.second_order = second_order
        self.dim = dim

        self.sensitivity_instance = sensitivity_instance
        self.check_instance = check_instance
        
        if self.dim == 1:
            check_ishigami(nb_parms, nb_sim, self.sensitivity_instance, self.check_instance, second_order = second_order)
        else :
            check_ishigami_multi_dim(nb_parms, nb_sim, self.sensitivity_instance, 
                                     self.check_instance, second_order = second_order,
                                     dim=dim)

    def check_firstorder(self):
        assert self.sensitivity_instance.getFirstOrderIndices().shape == (self.dim, self.nb_parms)

        if self.dim == 1 :
             assert np.allclose(self.check_instance.getFirstOrderIndices(), 
                            self.sensitivity_instance.getFirstOrderIndices(), atol=10e-10)
        else :
            sensitivity_i = self.sensitivity_instance.getFirstOrderIndices()
            for d in range(self.dim):
                assert np.allclose(self.check_instance[d].getFirstOrderIndices(), 
                                    sensitivity_i[d], atol=10e-10)

    def check_totalorder(self):
        assert self.sensitivity_instance.getTotalOrderIndices().shape == (self.dim, self.nb_parms)

        if self.dim == 1 :
            assert np.allclose(self.check_instance.getTotalOrderIndices(), 
                            self.sensitivity_instance.getTotalOrderIndices(), atol=10e-10)
        else :
            sensitivity_i = self.sensitivity_instance.getTotalOrderIndices()
            for i in range(self.dim):
                assert np.allclose(self.check_instance[i].getTotalOrderIndices(), 
                                    sensitivity_i[i], atol=10e-10)

    def check_secondorder(self):
        sensitivity_i = self.sensitivity_instance.getSecondOrderIndices()
        assert sensitivity_i.shape == (self.dim, self.nb_parms, self.nb_parms)

        if self.dim == 1 :
            np.allclose(self.check_instance.getSecondOrderIndices(), 
                                    sensitivity_i, atol=10e-10)
        else : 
            for d in range(self.dim):
                assert np.allclose(self.check_instance[d].getSecondOrderIndices(), 
                                    sensitivity_i[d], atol=10e-10)
        



class SensitivityTester_IshigamiOpenTurns:
    def __init__(self, nb_parms, nb_sim, dim, sensitivity_instance, method_name):
        self.nb_parms = nb_parms
        self.nb_sim = nb_sim
        self.second_order = False
        self.dim = dim

        self.sensitivity_instance = sensitivity_instance

        from tests.mock.sensitivity.check_ishigami import ishigami_with_openturns
        inputDesign, outputDesign = ishigami_with_openturns(nb_parms, nb_sim, sensitivity_instance)
        import openturns as ot

        if method_name == JANSEN:
            self.check_instance = ot.JansenSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        elif method_name == MARTINEZ:
            self.check_instance = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
        elif method_name == SALTELLI:
            self.check_instance = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, nb_sim)
    
    
    def check_firstorder(self):
        assert self.sensitivity_instance.getFirstOrderIndices().shape == (self.dim, self.nb_parms)
        assert np.allclose(self.check_instance.getFirstOrderIndices(), self.sensitivity_instance.getFirstOrderIndices()[0], atol=10e-10)

    def check_totalorder(self):
        assert self.sensitivity_instance.getTotalOrderIndices().shape == (self.dim, self.nb_parms)
        assert np.allclose(self.check_instance.getTotalOrderIndices(), self.sensitivity_instance.getTotalOrderIndices()[0], atol=10e-10)


