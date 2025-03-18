#!/bin/bash

source /nfs/pic.es/user/m/mlokken/.bashrc
conda activate /data/desi/common/pcross


# loop through 3 axes
for axis in {1..3}; do
    for sign in plus minus; do
        echo "Getting P1D for axis $axis with sign $sign"
        /data/desi/common/pcross/bin/python /nfs/pic.es/user/m/mlokken/p1d-px-fft-weight/masked_p1d_hydrosim.py $axis $sign dla_mask.npy
    done
done


# non_mask.npy dla_small_mask.npy random_mask.npy skyline_mask.npy double_skyline_mask.npy 