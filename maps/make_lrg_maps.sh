#!/bin/bash -l
#SBATCH -J LRG
#SBATCH -t 0:45:00
#SBATCH -N 4
#SBATCH -o MakeLRGmaps.out
#SBATCH -e MakeLRGmaps.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

sample=$1

module load evp-patch
conda activate cobaya
export OMP_NUM_THREADS=32
srun -n 32 -c 32 python make_lrg_maps.py ${sample}