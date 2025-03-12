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


class SpikeModel(object):
    """ Object to describe the contamination by a random spike in Fourier space"""
    def __init__(self, k_spike=0.5, a_spike = 0.5):
        """Power amplitude (Mpc) and scale (inverse Mpc) of the spike"""

        self.k_spike = k_spike
        self.a_spike = a_spike
    
    def power_contamination(self,k):
        """Add this to P1D to model the contamination by the spike"""

        a=self.a_spike
        is_closest_k = np.argmin(np.abs(k - self.k_spike))
        return_arr = np.zeros(len(k))
        return_arr[is_closest_k] = a
        return return_arr
        
    def contaminate_mode(self,k,delta_k,norm):
        """Input a Fourier mode k, the delta_k flux, and the normalization factor to convert to inverse Mpc units; return the contaminated mode
        If the mode is already normalized in the correct units, enter norm=1.
        delta_k should have shape [Nskew, Nk]"""

        a=self.a_spike / norm
        print(norm, a)
        is_closest_k = np.argmin(np.abs(k - self.k_spike))
        print(is_closest_k)
        print("delta k shape", delta_k.shape)
        eps = 1e-12
        is_neg_k = np.argmin(np.abs(k + self.k_spike))  # Find the corresponding negative mode
        print("contaminating factor is", np.sqrt(1 + a / (np.abs(delta_k[:,is_closest_k])**2 + eps)))
        new_delta_k = np.copy(delta_k)
        new_delta_k[:,is_closest_k] = delta_k[:,is_closest_k] * np.sqrt(1 + a / (np.abs(delta_k[:,is_closest_k])**2))
        new_delta_k[:,is_neg_k] = np.conj(new_delta_k[:,is_closest_k])  # Enforce Hermitian symmetry
        return new_delta_k
        