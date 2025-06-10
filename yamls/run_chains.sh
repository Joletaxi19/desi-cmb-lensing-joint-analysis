#!/bin/bash -l
#SBATCH -J run_chains
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -o log/run_chains.out
#SBATCH -e log/run_chains.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

name=$1

date
cosmodesienv 2025_03
export OMP_NUM_THREADS=8
if test -f "chains/${name}.1.txt"
then
    echo "Resuming chains with name: ${name}"
    srun -N 1 -n 16 -c 8 cobaya-run chains/${name}  
else
    echo "Starting a new chain with name: ${name}"
    srun -N 1 -n 16 -c 8 cobaya-run ${name}.yaml
fi
date
