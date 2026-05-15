#!/bin/bash -l

#SBATCH --job-name=wp_run
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --output=job_output/wp_output_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

cd /home/busato/assortative_partitions

module load gcc
module load python

source /home/busato/venvs/assortative_partitions/bin/activate

mkdir -p job_output

# ---- Default parameters (used when sbatch is called with no extra args) ----
DEFAULT_ARGS=(
    --K 3
    --d 5
    --H 3
    --problem-type assortative
    --eps 1e-3
    --damping 0.85
    --max-iter 500
    --tol 1e-8
    --init-type small_noise
    --m 1
    --n-obs-samples 20000
    --seed 0
    --device cpu
    --dtype float64
    --num-threads 8
    --wandb
    --log-every 10
    --save-dir results/wp
    --verbose 2
)

if [ "$#" -gt 0 ]; then
    EXTRA_ARGS=("$@")
else
    EXTRA_ARGS=("${DEFAULT_ARGS[@]}")
fi

RUN_ID=${SLURM_ARRAY_TASK_ID}
SEED=$((42 + RUN_ID))

srun --cpu-bind=cores \
    python /home/busato/assortative_partitions/src/warning_propagation.py \
        "${EXTRA_ARGS[@]}" \
        --seed ${SEED} \
        --run-name "wp_run_ass_K2_d8_H5_${RUN_ID}" \
        --wandb-name "wp_run_ass_K3_d3_H1_${RUN_ID}"