#!/bin/bash

#SBATCH --job-name=diverge
#SBATCH --time=1200
#SBATCH --account=enhancerg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=klebl@physik.rwth-aachen.de
#SBATCH --array=0-0

### GPU
#SBATCH --partition=dc-gpu

export OMP_NUM_THREADS=64

# SET 1
devices=( 0 0,1 0,1,2 0,1,2,3 )
ndevs=( 1 2 3 4 )
execs=( "./t2g_model.x 16 100" "./t2g_model.x 24 200" "./t2g_model.x 32 400" )
prefices=( "16_100_GPU" "24_200_GPU" "32_400_GPU" )

for ((dev=0; dev<${#devices[@]}; dev++)); do
for ((pre=0; pre<${#prefices[@]}; pre++)); do
    ndev=${ndevs[$dev]}
    executable=${execs[$pre]}
    prefix=${prefices[$pre]}
    mkdir -p $prefix

    fname=$(printf "$prefix/n%02d.dat" ${ndev})
    echo "# CUDA_VISIBLE_DEVICES=${devices[$dev]} LD_PRELOAD=./libdivERGe.so $executable" > $fname
    CUDA_VISIBLE_DEVICES=${devices[$dev]} LD_PRELOAD=./libdivERGe.so $executable >> $fname
done
done
