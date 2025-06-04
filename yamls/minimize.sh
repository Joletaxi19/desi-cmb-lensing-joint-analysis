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

cosmodesienv 2025_03
export OMP_NUM_THREADS=2
srun -n 1  -c 2 python minimize.py ${name}
if [[ ${name} == chains/* ]]
then
    srun -n 64 -c 2 cobaya-run --minimize --force ${name:7}_minimize.yaml
    srun -n 1  -c 2 rm ${name:7}_minimize.yaml
else
    srun -n 64 -c 2 cobaya-run --minimize --force ${name}_minimize.yaml
    srun -n 1  -c 2 rm ${name}_minimize.yaml
fi