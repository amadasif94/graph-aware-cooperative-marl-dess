#!/bin/bash
#SBATCH --job-name=eval_tagconv
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_tagconv_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_tagconv_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
TMP_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp"
PIP_CACHE_DIR_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache"

mkdir -p "${SLURM_LOG_DIR}" "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

export TMPDIR="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TMP="${TMP_BASE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_BASE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL="tagconv"
RUN_ID=${SLURM_ARRAY_TASK_ID}

CHECKPOINT_DIR="${BASE_DIR}/results/checkpoints/maddpg_ieee33/${MODEL}/${MODEL}_run_${RUN_ID}"

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
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
echo "START TIME           = $(date)"
echo "============================================================"

source /network/rit/lab/dahlin_lab/anaconda3/etc/profile.d/conda.sh
conda activate /network/rit/lab/dahlin_lab/amad/amad_thesis/env/thesis_conda

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "PYTHON = $(which python)"
python --version

cd "${BASE_DIR}"

echo "============================================================"
echo "Preflight CUDA/import test"
echo "============================================================"

python - <<PY
from pathlib import Path
import torch

ckpt_dir = Path("${CHECKPOINT_DIR}")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Checkpoint dir exists:", ckpt_dir.exists())
print("Checkpoint dir:", ckpt_dir)

if not ckpt_dir.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG

print("${MODEL} import test passed")
PY

echo "============================================================"
echo "Launching evaluation with MADDPG, zero, and random policies"
echo "============================================================"

shopt -s nullglob

for CKPT in "${CHECKPOINT_DIR}"/maddpg_step_*.pt
do
    STEP=$(basename "$CKPT" | sed 's/maddpg_step_//; s/.pt//')

    if [ "$STEP" -ge 15000 ] && [ "$STEP" -le "$MAX_STEP" ]; then

        RUN_NAME="eval/${MODEL}/run_${RUN_ID}/step_${STEP}"

        echo "------------------------------------------------------------"
        echo "Evaluating checkpoint:"
        echo "${CKPT}"
        echo "RUN_NAME = ${RUN_NAME}"
        echo "Policies = maddpg, zero, random"
        echo "------------------------------------------------------------"

        PYTHONPATH=. python -u scripts/run_eval_ieee33.py \
          --checkpoint "${CKPT}" \
          --gnn_type "${MODEL}" \
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
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit ${rc}