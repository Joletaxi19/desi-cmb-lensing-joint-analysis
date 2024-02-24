#!/bin/bash -l
#SBATCH -J mccorr
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

job=$1

conda activate noah_base
export OMP_NUM_THREADS=64
srun -N 1 -n 4 -c 64 python mc_corr_jobs.py ${job}