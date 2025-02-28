import numpy as np


class SiliconModel(object):
    """ Object to describe the contamination by SiIII(1207)"""

    def __init__(self,r_SiIII=20, a_SiIII=0.05):
        """Scale (in Mpc) and amplitude of the contamination"""

        # delta_Si(x) = a_SiIII * delta_alpha(x + r_SiIII)

        self.r_SiIII=r_SiIII
        self.a_SiIII=a_SiIII


    def power_contamination(self,k):
        """Multiply P1D by this to model the contamination by SiIII"""

        a=self.a_SiIII
        r=self.r_SiIII
        return 1 + a**2 + 2*a*np.cos(k*r)


    def mode_contamination(self,k):
        """Multiply a Fourier mode by this to add the contamination by SiIII"""

        a=self.a_SiIII
        r=self.r_SiIII
        return 1 + a * ( np.cos(k*r) - np.sin(k*r)*1j )
