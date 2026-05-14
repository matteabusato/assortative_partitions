#!/bin/bash

#SBATCH --job-name=sp_run
#SBATCH --output=logs/sp_%A_%a.out
#SBATCH --error=logs/sp_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu


set -euo pipefail

mkdir -p logs results/wp

# --- pick interpreter -------------------------------------------------------
PYTHON="${CLUSTER_PYTHON:-python}"

# --- parse a possible --configs flag ----------------------------------------
CONFIGS_FILE=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs) CONFIGS_FILE="$2"; shift 2 ;;
        *)         PASSTHROUGH+=("$1"); shift ;;
    esac
done

# --- decide what config to run ----------------------------------------------
if [[ -n "$CONFIGS_FILE" ]]; then
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "[error] --configs requires SLURM_ARRAY_TASK_ID (use --array=...)" >&2
        exit 2
    fi
    LINE_NO="$SLURM_ARRAY_TASK_ID"
    CONFIG_LINE="$(sed -n "${LINE_NO}p" "$CONFIGS_FILE")"
    if [[ -z "$CONFIG_LINE" ]]; then
        echo "[error] empty line $LINE_NO in $CONFIGS_FILE" >&2
        exit 2
    fi
    echo "[run_sp] line $LINE_NO from $CONFIGS_FILE: $CONFIG_LINE"
    # shellcheck disable=SC2086
    exec $PYTHON survey_propagation.py $CONFIG_LINE "${PASSTHROUGH[@]}"
else
    echo "[run_sp] direct args: ${PASSTHROUGH[*]}"
    exec $PYTHON survey_propagation.py "${PASSTHROUGH[@]}"
fi