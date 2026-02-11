#!/bin/bash -l

#SBATCH --job-name=bp_test_run
#SBATCH --partition=gpu                  # Use the GPU partition 
#SBATCH --gres=gpu:1   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=job_output/bp_output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

cd /home/busato/assortative_partitions

module load gcc
module load python
module load cuda

source /home/busato/venvs/assortative_partitions/bin/activate

srun python /home/busato/assortative_partitions/bp_3_assortative.py
