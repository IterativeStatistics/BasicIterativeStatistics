from typing import Dict 
import numpy as np
import copy

from iterative_stats.abstract_iterative_statistics import AbstractIterativeStatistics
from iterative_stats.iterative_mean import IterativeShiftedMean
from iterative_stats.utils.logger import logger 

class IterativeDotProduct(AbstractIterativeStatistics):
    def __init__(self, vector_size : int = 1, iterative_shifted_mean_1 : IterativeShiftedMean = None):
        """
            Compute iteratively the formula sum_k=0:N (A_k - shift_N)(B_k - shift_N)/(N-1)
            iterative_shifted_mean_1 (IterativeShiftedMean) : add an external iterative shifted mean. If it is not None, the mean will not be updated into this class
        """
        super().__init__(vector_size)

        if iterative_shifted_mean_1 is None :
            self.iterative_shifted_mean_1 = IterativeShiftedMean(vector_size)
            self.external_mean_1 = False
        else :
            self.iterative_shifted_mean_1 = iterative_shifted_mean_1
            self.external_mean_1 = True

        self.iterative_shifted_mean_2 = IterativeShiftedMean(vector_size)

        self.previous_shift = None
        self.collect_data = {'data_1' : [], 'data_2' : []}

    def increment(self, data_1, data_2, shift):
        self.iteration += 1
        
        if self.iteration == 1 :
            self.collect_data['data_1'] = np.append(self.collect_data['data_1'], data_1)
            self.collect_data['data_2']= np.append(self.collect_data['data_2'],data_2)

        elif self.iteration == 2 :
            self.collect_data['data_1'] = np.append(self.collect_data['data_1'], data_1)
            self.collect_data['data_2']= np.append(self.collect_data['data_2'],data_2)

            self.state = np.dot(self.collect_data['data_1'] - shift, self.collect_data['data_2'] - shift)
            del self.collect_data
        else :
            
            diff_shift = self.previous_shift - shift
            self.state *= (1 - 1/(self.iteration - 1))
            self.state += (data_1 - self.previous_shift)*(data_2 - self.previous_shift)/(self.iteration - 1)
            val = self.iterative_shifted_mean_1.get_stats() + self.iterative_shifted_mean_2.get_stats()
            val +=  diff_shift + (data_1 + data_2 - shift - self.previous_shift)/(self.iteration - 1)
            self.state += val * diff_shift
            
        # Update
        if not self.external_mean_1 :
            self.iterative_shifted_mean_1.increment(data_1, shift)

        self.iterative_shifted_mean_2.increment(data_2, shift)
        
        self.previous_shift = copy.deepcopy(shift)


    def get_mean_1(self):
        return self.iterative_shifted_mean_1.get_stats()

    def get_mean_2(self):
        return self.iterative_shifted_mean_2.get_stats()