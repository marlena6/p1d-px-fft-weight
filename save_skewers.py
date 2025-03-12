import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from mpi4py import MPI
import os
from hydrosim_skewers import read_and_save_spectra
from contamination import SiliconModel, SpikeModel

mode = 'delta_x'


sim_name=sys.argv[1]
# sim_name = 'diffSeed'
snap_num=int(sys.argv[2])
sim_dir  = sys.argv[3]
sk_fname="skewers_{}_Ns768_wM0.05_sT1.0_sg1.0.hdf5".format(snap_num)


delta_path_x = "/data/desi/common/HydroData/Emulator/post_768/delta_x"
delta_path_k = "/data/desi/common/HydroData/Emulator/post_768/delta_k"
if mode=='delta_k':
    try:
        os.mkdir(delta_path_k+f'/{sim_dir}')
    except:
        print("Save path already exists.")
    save_path_k = delta_path_k +f'/{sim_dir}'
    print(f"Will save delta_k information in {save_path_k}")
if mode=='delta_x':
    try:
        os.mkdir(delta_path_x+f'/{sim_dir}')
    except:
        print("Save path already exists.")
    save_path_x = delta_path_x +f'/{sim_dir}'
    print(f"Will save delta_x information in {save_path_x}")
    
data_path = "/data/desi/common/HydroData/Emulator/post_768/Australia20" # in PIC


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# size = 1
if size==1: # loop
    for phase in ["sim_plus", "sim_minus"]:
        if phase=="sim_plus":
            ind_phase = 0
        elif phase=="sim_minus":
            ind_phase = 1
        for axis in range(1,4):
            if mode=='delta_k':
                filesave = save_path_k+"/skewers_{}_p{}_ax{}_Ns768_wM0.05_sT1.0_sg1.0".format(snap_num, phase, axis)
            elif mode=='delta_x':
                filesave = save_path_x+"/skewers_{}_p{}_ax{}_Ns768_wM0.05_sT1.0_sg1.0".format(snap_num, phase, axis)
            if not os.path.exists(filesave+".npz"):
                print("Collecting spectra for phase {}, skewer {}".format(phase, axis))
                sk_dir="{}/{}/{}/skewers_{}".format(data_path,sim_name, phase, axis)
                Silicon = SiliconModel()
                Spike   = SpikeModel(a_spike=0.5)
                read_and_save_spectra(sk_dir, sk_fname, snap_num, axis, filesave, add_silicon=True, add_spike=True, mode=mode, Silicon=Silicon, Spike=Spike)
            else:
                print("Already exists.")

            
elif size==6:
    if rank < 3:
        phase = 'sim_plus'
        ind_phase = 0
    else:
        phase = 'sim_minus'
        ind_phase = 1
    axis = rank%3+1
    if mode=='delta_k':
        filesave = save_path_k+"/skewers_{}_p{}_ax{}_Ns768_wM0.05_sT1.0_sg1.0".format(snap_num, phase, axis)
    elif mode=='delta_x':
        filesave = save_path_x+"/skewers_{}_p{}_ax{}_Ns768_wM0.05_sT1.0_sg1.0".format(snap_num, phase, axis)
    if not os.path.exists(filesave):
        print("Collecting spectra for phase {}, skewer {}".format(phase, axis))
        sk_dir="{}/{}/{}/skewers_{}".format(data_path,sim_name, phase, axis)
        Silicon = SiliconModel()
        Spike   = SpikeModel(a_spike=0.5)
        read_and_save_spectra(sk_dir, sk_fname, snap_num, axis, filesave, add_silicon=True, add_spike=True, mode=mode, Silicon=Silicon, Spike=Spike)
