import numpy as np
from iterative_stats.utils.logger import logger
import copy
import time

def check_ishigami(nb_parms, nb_sim, sensitivity_indices, check_sensitivity, second_order: bool = False):
    from tests.mock.ishigami import ishigami 
    from tests.mock.uniform_3d import Uniform3D

    input_sample_generator = Uniform3D(nb_parms = nb_parms, nb_sim = nb_sim, second_order=second_order).generator()
    cpt = 0
    while True :
        logger.info(f'___________Round {cpt}________________')
        try :

            input_sample = next(input_sample_generator)
            output_sample = np.apply_along_axis(ishigami, 1,input_sample)
            sensitivity_indices.increment(output_sample)
            check_sensitivity.collect(output_sample)

            _ = check_sensitivity.getSecondOrderIndices()
            _ = check_sensitivity.getFirstOrderIndices()
        except StopIteration :
            # End of the test
            break 
        cpt += 1
        
def check_ishigami_multi_dim(nb_parms, nb_sim, sensitivity_indices, 
                                check_sensitivity : object = None, second_order: bool = False, 
                                dim : int = 1, with_timer: list = []):
    from tests.mock.ishigami import ishigami 
    from tests.mock.uniform_3d import Uniform3D

    input_sample_generator = [Uniform3D(nb_parms = nb_parms, nb_sim = nb_sim, second_order=second_order).generator() for _ in range(dim)]
    cpt = 0

    size_output = 2 + nb_parms
    if second_order :
        size_output += nb_parms
    while True :
        logger.info(f'___________Round {cpt}________________')
        try :
            outputs = np.zeros((size_output, dim))
            for i in range(dim) :
                input_sample = next(input_sample_generator[i])                
                outputs[:, i] = np.apply_along_axis(ishigami, 1,input_sample)
                if check_sensitivity is not None :
                    check_sensitivity[i].collect(copy.deepcopy(outputs[:,i]))
            if len(with_timer) > 0 :
                tic = time.perf_counter()
                sensitivity_indices.increment(outputs)
                toc = time.perf_counter()
                logger.info(f'Computation for a field of size {dim} {toc - tic:0.4f} seconds (n_sim = {nb_sim})')
                with_timer[cpt] = toc-tic
            else :
                sensitivity_indices.increment(outputs)
            # logger.info(f' second order: {sensitivity_indices.getSecondOrderIndices()}')
        except StopIteration :
            # End of the test
            break 
        cpt += 1

def ishigami_with_openturns(nb_parms, nb_sim, sensitivity_indices, check_sensitivity = None):
    import openturns as ot
    import copy 
    
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
        sensitivity_indices.increment(sample)
        if check_sensitivity is not None :
            check_sensitivity.collect(sample)

    return inputDesign, outputDesign

