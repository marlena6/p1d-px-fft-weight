import numpy as np
from contamination import SiliconModel, SpikeModel
from fake_spectra import spectra as spec
import matplotlib.pyplot as plt

def read_and_save_spectra(sk_dir, sk_fname, snap_num, axis, filesave, add_silicon=False, add_spike=False, mode='delta_k', Silicon=None, Spike=None, reduce_size=True):
    """ Function to read hydro simulation mock spectra and save them in np format with additional information.
    sk_dir (str): the directory of the saved mock spectra 
    filesave (str): the file to save into
    add_silicon (bool): whether or not to add mock silicon contamination
    mode (str): 'delta_x', 'delta_k'.
    """
    
    
    # reconstruct filename with skewers
    print("will load skewers from file {} in directory {}".format(sk_fname,sk_dir))
    # create Spectra object from fake_spectra repo
    spectra = spec.Spectra(num=snap_num,base="NA",cofm=None,axis=None,savedir=sk_dir,savefile=sk_fname,
                     res=None,reload_file=False,load_snapshot=False,quiet=False)
    if axis==1:
        axes = [1,2] # yz plane
    elif axis==2:
        axes = [0,2] # xz plane
    elif axis==3:
        axes = [0,1]
    delta_flux_x=delta(spectra.get_tau('H',1,1215))
    Ns,Np=delta_flux_x.shape
    print("number of pixels", Np)
    Lh = spectra.box/1000
    L = Lh/spectra.hubble
    print(f"Box is {L} Mpc per side")
    print(f"at redshift {spectra.red}")
    pix_spacing = L/Np
    print(f"Spacing between pixels along line-of-sight = {pix_spacing} Mpc")
    xpar = np.arange(0, L, pix_spacing)+pix_spacing/2.

    if mode=='delta_k' or (mode=='delta_x' and ((add_silicon is not None) or (add_spike is not None))):
        kpar = np.fft.fftfreq(Np, pix_spacing)*2*np.pi # frequency in Mpc^-1
        delta_flux_k=np.fft.fft(delta_flux_x) # note: changing from rfft to fft!!!
        if add_silicon:
            if Silicon is None:
                Silicon = SiliconModel() # initialize the default silicon model
            print(f"Adding mock silicon contamination with a={Silicon.a_SiIII} and r={Silicon.r_SiIII}.")
            # add the silicon model
            delta_flux_k *= Silicon.mode_contamination(kpar)
        if add_spike:
            print("Adding spike contamination.")
            if Spike is None:
                Spike = SpikeModel() # initialize the default spike model
            print(f"Adding random spike contamination with a={Spike.a_spike} and k={Spike.k_spike}.")
            # contaminate via the spike model
            delta_flux_k = Spike.contaminate_mode(kpar, delta_flux_k, norm=L/Np**2)
        if reduce_size:
            delta_flux_k = delta_flux_k.astype(np.complex64)
        kpar = kpar[:110]
        if mode=='delta_x':
            delta_flux_x = np.fft.ifft(delta_flux_k)
            print("Will save contaminated version of the delta(x) fluxes")
            print("Shape will be:", delta_flux_x.shape)
        # limit to kpar up to ~10 Mpc-1
        delta_flux_k=(delta_flux_k[:,:110])
        
    
    print(f"Saving to {filesave}.")
    if mode=='delta_k':
        np.savez(filesave, L_Mpc=L, los_spacing_Mpc=pix_spacing, delta_flux_k=delta_flux_k*np.sqrt(pix_spacing/Np), kpar=kpar) # normalize the delta_k before saving
    if mode=='delta_x':
        np.savez(filesave, L_Mpc=L, los_spacing_Mpc=pix_spacing, delta_flux_x=delta_flux_x, xpar=xpar)
    return

def delta(tau):
    # compute delta flux given optical depth tau
    flux = np.exp(-tau)
    # compute mean flux
    mean_flux = np.mean(flux)
    # and Lya fluctuations
    delta_flux = flux/mean_flux-1.0
    return (delta_flux)


def bin_spectra(spectra, x_spectra, bin_size):
    """
    Bin spectra over the Np axis.

    Parameters:
    spectra (np.ndarray): The input 2D array of shape (Ns, Np).
    x_spectra (np.ndarray): The input 1D array of shape (Np).
    bin_size (int): The number of adjacent pixels to combine for binning.

    Returns:
    np.ndarray: The binned spectra array.
    np.ndarray: The binned x_spectra array.
    """
    Ns, Np = spectra.shape
    # Calculate the number of bins
    num_bins = Np / bin_size
    if num_bins % 1 != 0:
        raise ValueError("bin_size must evenly divide Np.")
        # # Truncate the spectra to make it evenly divisible by bin_size
        # spectra = spectra[:, :num_bins * bin_size]
    else:
        num_bins = int(num_bins)
        print("Number of bins:", num_bins)
    # Reshape and sum/average over the new axis
    binned_spectra = spectra.reshape(Ns, num_bins, bin_size).mean(axis=2)
    binned_x_spectra = x_spectra.reshape(num_bins, bin_size).mean(axis=1)

    return binned_spectra, binned_x_spectra
