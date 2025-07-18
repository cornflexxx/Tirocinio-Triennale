#!/bin/bash
#SBATCH --job-name=hom_all
#SBATCH --output=out.log
#SBATCH --error=err.log
#SBATCH --time=00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --account=IscrC_ASCEND

module purge
module load cuda
module load openmpi
make  main_allred

srun ./main_allred 4
