#!/bin/bash
echo "Starting script"
source /nfs/pic.es/user/m/mlokken/.bashrc
conda activate /data/desi/scratch/mlokken/lace
echo "Activated."
# set sim_name
sim_name='sim_pair_30' # other version will be 'diffSeed'
sim_dir='central_with_silicon_spike'

# loop through snapshots from 0 to 10
for snapnum in {0..10}
do
    echo "Making skewer files for snapnum $snapnum"
    # make the skewer delta_k files
    mpirun -np 6 /data/desi/scratch/mlokken/lace/bin/python /nfs/pic.es/user/m/mlokken/p1d-px-fft-weight/save_skewers.py $sim_name $snapnum $sim_dir 
done