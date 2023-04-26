import pytest 

from iterative_stats.sensitivity.sensitivity_jansen import IterativeSensitivityJansen as IterativeSensitivityMethod
from iterative_stats.utils.logger import logger
from tests.mock.sensitivity.check_jansen import JansenCheckSensitivityIndices as SensitivityIndicesChecker

NB_ISHIGAMI_PARMS = 3

# COMPARISON WITH NON-ITERATIVE FORMULAE
from tests.unit.sensitivity.template_testsensitivity import SensitivityTester_Ishigami as SensitivityTester
@pytest.fixture
def tester():
    # Create a closure on the Tester object
    def _tester(nb_parms, nb_sim, second_order, dim):
        sensitivity_instance = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = second_order)
        if dim == 1 :
            check_instance = SensitivityIndicesChecker(nb_parms = nb_parms, second_order = second_order)
        else :
            check_instance = [SensitivityIndicesChecker(nb_parms = nb_parms, second_order = second_order) for _ in range(dim)]
        return SensitivityTester(nb_parms, nb_sim, second_order, dim, sensitivity_instance,check_instance)
    
    # Pass this closure to the test
    yield _tester 

@pytest.mark.parametrize(['nb_parms', 'nb_sim', 'second_order', 'dim'], [[NB_ISHIGAMI_PARMS,10,True, 1]])
def test_jansen_ishigami(tester, nb_parms, nb_sim, second_order, dim):
    # run the closure now with the desired params
    my_tester = tester(nb_parms, nb_sim, second_order, dim)

    my_tester.check_firstorder()
    my_tester.check_totalorder()
    my_tester.check_secondorder()

@pytest.mark.parametrize(['nb_parms', 'nb_sim', 'second_order', 'dim'], [[NB_ISHIGAMI_PARMS,3,False, 5]])
def test_jansen_ishigami_multidim(tester, nb_parms, nb_sim, second_order, dim):
    # run the closure now with the desired params
    my_tester = tester(nb_parms, nb_sim, second_order, dim)

    my_tester.check_firstorder()
    my_tester.check_totalorder()
    # my_tester.check_secondorder()

# COMPARISON WITH OPENTURNS
from tests.unit.sensitivity.template_testsensitivity import SensitivityTester_IshigamiOpenTurns
@pytest.fixture
def tester_openturns():
    import openturns as ot
    # Create a closure on the Tester object
    def _tester(nb_parms, nb_sim, dim):
        sensitivity_instance = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = False)
        return SensitivityTester_IshigamiOpenTurns(nb_parms, nb_sim, dim, sensitivity_instance, method_name= 'Jansen')
    
    # Pass this closure to the test
    yield _tester 


@pytest.mark.parametrize(['nb_parms', 'nb_sim', 'dim'], [[NB_ISHIGAMI_PARMS,10,1]])
def test_jansen_ishigami_ot(tester_openturns, nb_parms, nb_sim, dim):
    # run the closure now with the desired params
    my_tester = tester_openturns(nb_parms, nb_sim, dim)

    my_tester.check_firstorder()
    my_tester.check_totalorder()





            
    # def test_ishigami_multi_dim(self):
    #     nb_parms = 3
    #     nb_sim = 3
    #     second_order = False
    #     dim = 5
    #     sensitivity_indices = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = second_order)
    #     check_sensitivity = [SensitivityIndicesChecker(nb_parms = nb_parms, second_order = second_order) for _ in range(dim)]
    #     check_ishigami_multi_dim(nb_parms, nb_sim, sensitivity_indices, 
    #                                 check_sensitivity, second_order = second_order,
    #                                 dim=dim)

    #     iterative_firstorderindices = sensitivity_indices.getFirstOrderIndices()
    #     # Check output size
    #     self.assertTupleEqual(iterative_firstorderindices.shape, (dim, nb_parms))

    #     if second_order :
    #         iterative_secondorderindices = sensitivity_indices.getSecondOrderIndices()
    #         logger.info(f'iterative second order (all) : {iterative_secondorderindices}')
    #     iterative_totalorderindices = sensitivity_indices.getTotalOrderIndices()
    #     # Check output size
    #     self.assertTupleEqual(iterative_totalorderindices.shape, (dim, nb_parms))


    #     for i in range(dim):
    #         check_firstorderindices = check_sensitivity[i].compute_firstorderindices()
    #         check_totalorderindices = check_sensitivity[i].compute_totalorderindices()

    #         if second_order:
    #             check_secondorderindices = check_sensitivity[i].compute_secondorderindices()

    #         for p in range(nb_parms):
    #             # check first order
    #             logger.info(f' {check_totalorderindices} {iterative_totalorderindices}')
    #             self.assertTrue(np.allclose(check_firstorderindices[p], iterative_firstorderindices[p,i], atol=10e-10))
    #             # check second order
    #             if second_order:
    #                 logger.info(f'(gt) {check_secondorderindices[p]}')
    #                 logger.info(f'second order {iterative_secondorderindices[i,p]}')
    #                 self.assertTrue(np.allclose(check_secondorderindices[p], iterative_secondorderindices[i,p], atol=10e-10))
    #             # check total order
    #             self.assertTrue(np.allclose(check_totalorderindices[p], iterative_totalorderindices[p,i], atol=10e-10))


 

