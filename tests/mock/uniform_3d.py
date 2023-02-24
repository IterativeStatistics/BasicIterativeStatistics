from scipy.stats import qmc

from experimental_design.experiment import AbstractExperiment

class Uniform3D(AbstractExperiment):
    """
        Draw a 3D uniform law based on the Latin Hypercube Sampling method (LHS).
        Use the qmc method from scipy
    """
    def __init__(self, l_bounds: list = None, u_bounds: list = None, **kwargs):
        super().__init__(**kwargs)
        self.sampler = qmc.LatinHypercube(d=self.nb_parms)
        self.l_bounds = l_bounds
        self.u_bounds = u_bounds

    def draw(self):
        sample = self.sampler.random(n=1)
        if self.l_bounds is None or self.u_bounds is None :
            return sample
        else :
            return qmc.scale(sample, self.l_bounds, self.u_bounds)
        


