import numpy as np

from iterative_stats.sensitivity.abstract_sensitivity import IterativeAbstractSensitivity
from iterative_stats.iterative_variance import IterativeVariance
from iterative_stats.iterative_covariance import IterativeCovariance
from iterative_stats.utils.logger import logger


class IterativeSensitivityMartinez(IterativeAbstractSensitivity):
    """
    Estimates the Sobol indices iteratively based on the Martinez formula.
    """
    def __init__(self, nb_parms: int, dim: int = 1, second_order: bool = False, state: object = None):
        super().__init__(nb_parms = nb_parms, dim = dim, second_order=second_order, state=state)
        if state is None:
            
            self.var_B = IterativeVariance(dim)
            self.var_E = [IterativeVariance(dim) for _ in range(nb_parms)]
        
            self.covData_AE = [IterativeCovariance(dim) for _ in range(nb_parms)]
            self.covData_BE = [IterativeCovariance(dim) for _ in range(nb_parms)]
        
            self.pearson_A = np.zeros((dim, nb_parms))
            self.pearson_B = np.zeros((dim, nb_parms))
        
            del self.state
       
    def _increment(self, data):

        sample_A = data[0]
        sample_B = data[1]
        sample_E = data[2:(2 + self.nb_parms)]
        self.var_B.increment(sample_B)

        for p in range(self.nb_parms):
            self.var_E[p].increment(sample_E[p])

            # update first order
            self.covData_BE[p].increment(sample_B,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_B.get_stats())
            self.pearson_B[:,p] = np.divide(self.covData_BE[p].get_stats(), np.sqrt(var_prod))

            # update last order
            self.covData_AE[p].increment(sample_A,sample_E[p])
            var_prod = np.multiply(self.var_E[p].get_stats(), self.var_A.get_stats())
            if var_prod.any() > 0 :
                self.pearson_A[:,p] = np.divide(self.covData_AE[p].get_stats(), np.sqrt(var_prod))
           

    def getFirstOrderIndices(self) :
        return self.pearson_B

    def getTotalOrderIndices(self) :
        return 1. - self.pearson_A

    def _compute_varianceI(self):
        res = np.zeros((self.dimension, self.nb_parms))
        for p in range(self.nb_parms):
            res[:,p] = np.multiply(self.pearson_B[:,p],self.var_A.get_stats())
        return res

    def _compute_VTi(self):
        return None

    def save_state(self):
        """
            An abstract method to implement. It save the current state of the objects.
        """
        state = super().save_state()
        state['var_B'] = self.var_B.save_state()
        state['var_E'] = [self.var_E[i].save_state() for i in range(self.nb_parms)]
        state['covData_AE'] = [self.covData_AE[i].save_state() for i in range(self.nb_parms)]
        state['covData_BE'] = [self.covData_BE[i].save_state() for i in range(self.nb_parms)]
        state['pearson_A'] = self.pearson_A
        state['pearson_B'] = self.pearson_B
        return state

    def load_from_state(self, state: object):
        """
            It load the current state of the object.
        """
        super().load_from_state(state)
        self.var_B = IterativeVariance(self.dimension, state=state.get('var_B', None))
        self.var_E= [IterativeVariance(self.dimension, state=s) for s in state.get('var_E', None)]
        
        self.covData_AE = [IterativeCovariance(self.dimension, state=s) for s in state.get('covData_AE', None)]
        self.covData_BE = [IterativeCovariance(self.dimension, state=s) for s in state.get('covData_BE', None)]
        
        self.pearson_A = state.get('pearson_A', None)
        self.pearson_B = state.get('pearson_B', None)
