#!/bin/bash -l
#SBATCH -J minimize
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -o log/minimize.out
#SBATCH -e log/minimize.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

name=$1

date
module load evp-patch
conda activate cobaya
export OMP_NUM_THREADS=32
srun -n 1 -c 32 python minimize.py ${name}
srun -n 1 -c 32 cobaya-run --minimize --force ${name}_minimize.yaml
srun -n 1 -c 32 rm ${name}_minimize.yaml