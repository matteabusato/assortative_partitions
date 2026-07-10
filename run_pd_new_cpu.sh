#!/bin/bash -l

#SBATCH --job-name=pd_new_run
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=job_output/pd_new_output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

cd /home/busato/assortative_partitions

module load gcc
module load python

source /home/busato/venvs/assortative_partitions/bin/activate

srun --cpu-bind=cores python /home/busato/assortative_partitions/src/run_k2_assortative_sweep.py