import numpy as np
from iterative_stats.utils.logger import logger

def check_ishigami(nb_parms, nb_sim, sensitivity_indices, check_sensitivity):
    from tests.mock.ishigami import ishigami 
    from tests.mock.uniform_3d import Uniform3D

    input_sample_generator = Uniform3D(nb_parms = nb_parms, nb_sim = nb_sim).generator()

    while True :
        try :
            input_sample = next(input_sample_generator)
            output_sample = np.apply_along_axis(ishigami, 1,input_sample)
            sensitivity_indices.increment(output_sample)
            check_sensitivity.collect(output_sample)

            _ = check_sensitivity.compute_firstorderindices()
        except StopIteration :
            # End of the test
            break 


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

    logger.info(f'inputDesign : {inputDesign}')
    logger.info(f'outputDesign : {outputDesign}')
    return inputDesign, outputDesign

