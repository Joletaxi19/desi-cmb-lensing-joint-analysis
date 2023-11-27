#!/bin/bash -l
#SBATCH -J force
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -o log/force_restart.out
#SBATCH -e log/force_restart.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

name=$1

date
module load evp-patch
conda activate cobaya
export OMP_NUM_THREADS=8
echo "Restarting chain with name: ${name}"
srun -N 1 -n 16 -c 8 cobaya-run --force ${name}.yaml
date
