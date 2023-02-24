import numpy as np

def ishigami(x) :
    x = x * np.pi
    return np.sin(x[0]) + 7 * np.sin(x[1])**2 + 0.1 *np.sin(x[0]) *x[2]**4
