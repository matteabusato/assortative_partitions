#!/bin/bash -l

#SBATCH --job-name=bp_test_run
#SBATCH --partition=standard 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=job_output/bp_output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

cd /home/busato/assortative_partitions

module load gcc
module load python

source /home/busato/venvs/assortative_partitions/bin/activate

srun --cpu-bind=cores python /home/busato/assortative_partitions/bp_3_assortative.py

echo "Job $SLURM_JOB_ID completed successfully!" | mail -s "Job finished" mattea.busato@epfl.ch
