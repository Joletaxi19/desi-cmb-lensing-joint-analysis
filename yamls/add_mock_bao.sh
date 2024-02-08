#!/bin/bash -l
#SBATCH -J addMockBAO
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -o log/addMockBAO.out
#SBATCH -e log/addMockBAO.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi

name=$1

date
module load evp-patch
conda activate cobaya
export OMP_NUM_THREADS=8
srun -n 1  -c 8 python add_mock_bao.py ${name}
srun -n 16 -c 8 cobaya-run --force add_mock_bao_${name}.yaml
srun -n 1  -c 8 rm add_mock_bao_${name}.yaml