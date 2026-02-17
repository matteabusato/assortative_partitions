#!/bin/bash -l

#SBATCH --job-name=hardcoded_test_run
#SBATCH --partition=standard 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=job_output/hardcoded_output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

cd /home/busato/assortative_partitions

module load gcc
module load python

source /home/busato/venvs/assortative_partitions/bin/activate

srun --cpu-bind=cores python /home/busato/assortative_partitions/hardcoded_solver_3_assortative.py

