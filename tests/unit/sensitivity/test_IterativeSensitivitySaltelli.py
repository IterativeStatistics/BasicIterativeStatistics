import pytest 

from iterative_stats.sensitivity.sensitivity_saltelli import IterativeSensitivitySaltelli as IterativeSensitivityMethod
from iterative_stats.sensitivity import SALTELLI
from iterative_stats.utils.logger import logger
from tests.mock.sensitivity.check_saltelli import SaltelliCheckSensitivityIndices as SensitivityIndicesChecker

NB_ISHIGAMI_PARMS = 3
NB_SIM = 2


# COMPARISON WITH NON-ITERATIVE FORMULAE
from tests.unit.sensitivity.template_testsensitivity import SensitivityTester_Ishigami as SensitivityTester
@pytest.fixture
def tester():
    def _tester(nb_parms, nb_sim, second_order, dim):
        sensitivity_instance = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = second_order)
        if dim == 1 :
            check_instance = SensitivityIndicesChecker(nb_parms = nb_parms, second_order = second_order)
        else :
            check_instance = [SensitivityIndicesChecker(nb_parms = nb_parms, second_order = second_order) for _ in range(dim)]
        return SensitivityTester(nb_parms, nb_sim, second_order, dim, sensitivity_instance,check_instance)
    
    yield _tester


@pytest.mark.parametrize(['nb_parms', 'nb_sim', 'second_order', 'dim'], [[NB_ISHIGAMI_PARMS,NB_SIM, False, 1]])
def test_saltelli_ishigami(tester, nb_parms, nb_sim, second_order, dim):
    # run the closure now with the desired params
    my_tester = tester(nb_parms, nb_sim, second_order, dim)
    
    my_tester.check_totalorder()
    my_tester.check_firstorder()
    
    # my_tester.check_secondorder()

@pytest.mark.parametrize(['nb_parms', 'nb_sim', 'second_order', 'dim'], [[NB_ISHIGAMI_PARMS,NB_SIM,False, 5]])
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
    def _tester(nb_parms, nb_sim, dim):
        sensitivity_instance = IterativeSensitivityMethod(dim = dim, nb_parms = nb_parms, second_order = False)
        return SensitivityTester_IshigamiOpenTurns(nb_parms, nb_sim, dim, sensitivity_instance, method_name= SALTELLI)
    yield _tester 


# @pytest.mark.parametrize(['nb_parms', 'nb_sim', 'dim'], [[NB_ISHIGAMI_PARMS,NB_SIM,1]])
# def test_saltelli_ishigami_ot(tester_openturns, nb_parms, nb_sim, dim):
#     # run the closure now with the desired params
#     my_tester = tester_openturns(nb_parms, nb_sim, dim)

#     my_tester.check_firstorder()
#     my_tester.check_totalorder()
