#!/bin/bash -l

#SBATCH --job-name=k2-ass-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-42%4
#SBATCH --output=job_output/k2_full_%A_%a.out
#SBATCH --error=job_output/k2_full_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mattea.busato@epfl.ch

set -euo pipefail

cd /home/busato/assortative_partitions
mkdir -p job_output results/k2_full_sweep

module purge
module load gcc/11.3.0
module load cuda/11.8.0

source /home/busato/venvs/assortative_partitions/bin/activate

export WANDB_MODE=online
export WANDB_DIR=/home/busato/assortative_partitions/wandb
mkdir -p "$WANDB_DIR"

# Optional overrides:
export PD_SEED="${PD_SEED:-42}"
export PD_M="${PD_M:-1000000}"
export PD_MAX_ITER="${PD_MAX_ITER:-20000}"
export PD_OBS_FACTOR="${PD_OBS_FACTOR:-100}"
export PD_MIN_OBS_SAMPLES="${PD_MIN_OBS_SAMPLES:-20}"

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")
print("GPU:", torch.cuda.get_device_name(0))
PY

srun --cpu-bind=cores \
    python /home/busato/assortative_partitions/src/run_k2_assortative_full_sweep.py
