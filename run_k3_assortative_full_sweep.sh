#!/bin/bash -l

#SBATCH --job-name=k3-ass-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=job_output/k3_ass_full_%A_%a.out
#SBATCH --error=job_output/k3_ass_full_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

set -euo pipefail

PROJECT_ROOT="/home/busato/assortative_partitions"
cd "${PROJECT_ROOT}"

mkdir -p job_output
mkdir -p results/k3_assortative_full_sweep

module purge
module load gcc/11.3.0
module load python/3.10.4
module load cuda/11.8.0

source /home/busato/venvs/assortative_partitions/bin/activate

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Core population-dynamics settings.
export PD_M="${PD_M:-1000000}"
export PD_SEED="${PD_SEED:-43}"
export PD_MAX_ITER="${PD_MAX_ITER:-20000}"
export PD_MPARISI="${PD_MPARISI:-1.0}"
export PD_DAMPING="${PD_DAMPING:-0.8}"
export PD_TOL="${PD_TOL:-0.015}"
export PD_CONVERGENCE_CHECK_EVERY="${PD_CONVERGENCE_CHECK_EVERY:-200}"
export PD_STABLE_CHECKS="${PD_STABLE_CHECKS:-5}"
export PD_INIT_TYPE="${PD_INIT_TYPE:-hard_field}"

# Observable sampling.
# With M=1,000,000 and PD_OBS_FACTOR=10, each observable round uses
# 10,000,000 population-message draws, processed in GPU batches.
export PD_OBS_FACTOR="${PD_OBS_FACTOR:-10}"
export PD_OBSERVABLE_BATCH_SIZE="${PD_OBSERVABLE_BATCH_SIZE:-20000}"
export PD_MIN_OBS_SAMPLES="${PD_MIN_OBS_SAMPLES:-20}"
export PD_SAMPLING_START="${PD_SAMPLING_START:-5000}"
export PD_SAMPLING_INTERVAL="${PD_SAMPLING_INTERVAL:-500}"
export PD_REQUIRE_CONVERGENCE_FOR_SAMPLING="${PD_REQUIRE_CONVERGENCE_FOR_SAMPLING:-true}"

# Leave PD_NUM_SAMPLES unset to use PD_OBS_FACTOR * M.
# Define it explicitly only to override the upsampling rule.

# Optional intermediate diagnostics. Disabled by default for the full sweep.
export PD_SAVE_DIAGNOSTIC_PLOTS="${PD_SAVE_DIAGNOSTIC_PLOTS:-false}"
export PD_DIAGNOSTIC_HIST_BINS="${PD_DIAGNOSTIC_HIST_BINS:-80}"
export PD_DIAGNOSTIC_SAMPLE_SIZE="${PD_DIAGNOSTIC_SAMPLE_SIZE:-100000}"

export PD_USE_WANDB="${PD_USE_WANDB:-true}"
export PD_OUTPUT_DIR="${PD_OUTPUT_DIR:-${PROJECT_ROOT}/results/k3_assortative_full_sweep}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "Host=$(hostname)"
echo "Start=$(date --iso-8601=seconds)"

srun --cpu-bind=cores \
    python -u src/run_k3_assortative_full_sweep.py

echo "End=$(date --iso-8601=seconds)"