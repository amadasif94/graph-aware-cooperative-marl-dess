#!/bin/bash
#SBATCH --job-name=eval_mlp
#SBATCH --partition=batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_mlp_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_mlp_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
TMP_BASE="${BASE_DIR}/tmp"
PIP_CACHE_DIR_BASE="${BASE_DIR}/pip_cache"

CONDA_SH="/network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="/network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda"

mkdir -p "${SLURM_LOG_DIR}" "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

JOB_TMP="${TMP_BASE}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
JOB_PIP_CACHE="${PIP_CACHE_DIR_BASE}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${JOB_TMP}" "${JOB_PIP_CACHE}"

export TMPDIR="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export TMP="${JOB_TMP}"
export PIP_CACHE_DIR="${JOB_PIP_CACHE}"
export RAY_TMPDIR="${JOB_TMP}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

MODEL="mlp"
RUN_ID="${SLURM_ARRAY_TASK_ID}"

CHECKPOINT_DIR="${BASE_DIR}/results/checkpoints/maddpg_mlp_ieee33/mlp_run_${RUN_ID}"

if [ "${RUN_ID}" -eq 0 ]; then
    MAX_STEP=100000
else
    MAX_STEP=200000
fi

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = ${MODEL}"
echo "RUN_ID               = ${RUN_ID}"
echo "CHECKPOINT_DIR       = ${CHECKPOINT_DIR}"
echo "MAX_STEP             = ${MAX_STEP}"
echo "HOSTNAME             = $(hostname)"
echo "START TIME           = $(date)"
echo "CPUS REQUESTED       = ${SLURM_CPUS_PER_TASK}"
echo "TMPDIR               = ${TMPDIR}"
echo "============================================================"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}" || {
    echo "ERROR: conda activate failed"
    exit 127
}

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

cd "${BASE_DIR}"

echo "============================================================"
echo "Preflight import/checkpoint-dir test"
echo "============================================================"

python - <<PY
from pathlib import Path
import torch

ckpt_dir = Path("${CHECKPOINT_DIR}")

print("CUDA available:", torch.cuda.is_available())
print("Checkpoint dir exists:", ckpt_dir.exists())
print("Checkpoint dir:", ckpt_dir)

if not ckpt_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

from environments.dess_env import DESSEnv

print("Import test passed")
PY

echo "============================================================"
echo "Launching MLP evaluation with MADDPG, zero, and random policies"
echo "============================================================"

shopt -s nullglob

for CKPT in "${CHECKPOINT_DIR}"/maddpg_step_*.pt
do
    STEP=$(basename "${CKPT}" | sed 's/maddpg_step_//; s/.pt//')

    if [ "${STEP}" -ge 15000 ] && [ "${STEP}" -le "${MAX_STEP}" ]; then

        RUN_NAME="eval/${MODEL}/run_${RUN_ID}/step_${STEP}"

        echo "------------------------------------------------------------"
        echo "Evaluating checkpoint:"
        echo "${CKPT}"
        echo "RUN_NAME = ${RUN_NAME}"
        echo "Policies = maddpg, zero, random"
        echo "------------------------------------------------------------"

        PYTHONPATH=. python -u experiments/no_gnn_maddpg/run_eval_mlp_ieee33.py \
          --checkpoint "${CKPT}" \
          --mode test \
          --episodes 10 \
          --include_zero \
          --include_random \
          --run_name "${RUN_NAME}"

    fi
done

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "EXIT CODE     = ${rc}"
echo "============================================================"

exit ${rc}