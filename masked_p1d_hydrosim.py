# run with environment /data/desi/common/pcross
import numpy as np
import sys
import os
import hydrosim_skewers as hs

# set simulation details
sim_name='sim_pair_30' # other version will be 'diffSeed'
sim_dir='central_with_silicon_spike'
snap_num = 6
axis = sys.argv[1]
phase = sys.argv[2]
maskfile1 = sys.argv[3]
maskfile2 = None
maskfile3 = None
maskfile4 = None
maskfile5 = None
maskfile6 = None
if len(sys.argv)>4:
    maskfile2 = sys.argv[4]
if len(sys.argv)>5:
    maskfile3 = sys.argv[5]
if len(sys.argv)>6:    
    maskfile4 = sys.argv[6]
if len(sys.argv)>7:    
    maskfile5 = sys.argv[7]
if len(sys.argv)>7:    
    maskfile6 = sys.argv[8]
system='PIC'
if system=='PIC':
    meas_path = "/data/desi/common/HydroData/Emulator/post_768/"
    data_filepath = meas_path+"/delta_x/{:s}/skewers_{:d}_psim_{:s}_ax{:s}_Ns768_wM0.05_sT1.0_sg1.0.npz".format(sim_dir, snap_num, phase, axis)
    delta_x = np.load(data_filepath)
elif system=='home':
    meas_path = "/Users/mlokken/research/lyman_alpha/data/"
    data_filepath = meas_path+"snap_{:d}/delta_x/skewers_{:d}_psim_{:s}_ax{:s}_Ns768_wM0.05_sT1.0_sg1.0.npz".format(snap_num, phase, axis)
    delta_x = np.load(meas_path+"snap_{:d}/delta_x/skewers_{:d}_psim_{:s}_ax{:s}_Ns768_wM0.05_sT1.0_sg1.0.npz".format(snap_num, phase, axis))



print("Sim dir is:", sim_dir)
# read the masks
if 'silicon' in sim_dir:
    print("adding.")
    addon = '_silicon'
if 'spike' in sim_dir:
    addon += '_spike'
else:
    addon = ''
print(addon)

maskdir = f"/data/desi/scratch/mlokken/masking{addon}/"

print("Reading data from", data_filepath) 
delta_flux_x = delta_x['delta_flux_x'] # these are already normalized
Np = delta_flux_x.shape[1]
L = delta_x['L_Mpc']
pix_spacing = L/Np
xpar = delta_x['xpar'] # this is still wrong for some reason, redefined below
delta_x.close()
# xpar = np.arange(0, L, pix_spacing)+pix_spacing/2.

nskew = delta_flux_x.shape[0]
nside = np.sqrt(nskew).astype(int)
print(f"Box is {L} Mpc per side with {nside} skewers per side")
print(f"Spacing between pixels along line-of-sight = {pix_spacing} Mpc")
bin = False
if bin:
    # bin delta_flux along line-of-sight
    binsize = 5
    if Np%binsize != 0:
        print("Np is not divisible by binsize")
        sys.exit()
    delta_flux_x, xpar = hs.bin_spectra(delta_flux_x, xpar, binsize)
    Np=delta_flux_x.shape[1]
    pix_spacing = xpar[1]-xpar[0]
    print(f"Spacing between pixels along line-of-sight after binning = {pix_spacing} Mpc")

kpar = np.fft.fftfreq(Np, pix_spacing)*2*np.pi # frequency in Mpc^-1
# get the positions
xpos = np.linspace(0,L,nside)
ypos = np.linspace(0,L,nside)
print("spacing between neighboring skewers is {:.2f} Mpc".format(xpos[1]-xpos[0]))
positions = np.array([[x,y] for x in xpos for y in ypos])


for i,maskfile in enumerate([maskfile1, maskfile2, maskfile3, maskfile4, maskfile5, maskfile6]):
    if maskfile is not None:
        mask=np.load(maskdir+maskfile)
        maskname=os.path.splitext(maskfile)[0]
        savefile = maskdir+"skewers_{:d}_psim_{:s}_ax{:s}".format(snap_num, phase, axis)+ maskname+"_p1d"
        print(f"Checking to see if {savefile} exists")
        if not os.path.exists(savefile+".npz"):
            delta_flux_masked = delta_flux_x * mask
            delta_flux_masked_k = (np.fft.fft(delta_flux_masked, axis=1))
            # delete the pixel-space arrays to save memory
            del delta_flux_masked
            p1d_masked = np.mean((delta_flux_masked_k.__abs__())**2, axis=0)            
            np.savez(savefile, p1d_masked_dimless=p1d_masked, L_Mpc=L, los_spacing_Mpc=pix_spacing, kpar=kpar,  Np=Np, Nskew=nskew)

        else:
            print("Already exists, moving on.")