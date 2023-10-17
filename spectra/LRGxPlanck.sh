#!/bin/bash -l
#SBATCH -J comp_spec
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -o log/compute_spectra.out
#SBATCH -e log/compute_spectra.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi
date
module load evp-patch
conda activate noah_base
export OMP_NUM_THREADS=32
srun -N 1 -n 1 -c 32 python LRGxPlanck.py 4
date