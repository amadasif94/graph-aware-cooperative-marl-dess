#!/bin/bash
#SBATCH --job-name=eval_tp_gat_100k
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_tp_gat_100k_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/eval_tp_gat_100k_%A_%a.err

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

RUN_NAME="gat_tp_full_eval_100k"
CHECKPOINT_PATH="${BASE_DIR}/results/checkpoints/maddpg_ieee33/gat/gat_run_0/maddpg_step_100000.pt"

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "MODEL                = MADDPG-GAT"
echo "RUN_NAME             = ${RUN_NAME}"
echo "CHECKPOINT_PATH      = ${CHECKPOINT_PATH}"
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
echo "Preflight CUDA/import/checkpoint test"
echo "============================================================"
python - <<PY
from pathlib import Path
import torch

ckpt = Path("${CHECKPOINT_PATH}")

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("Checkpoint exists:", ckpt.exists())
print("Checkpoint path:", ckpt)

if not ckpt.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG

print("GAT import/checkpoint test passed")
PY

echo "============================================================"
echo "Launching GAT topology evaluation at 100000 steps"
echo "============================================================"

PYTHONPATH=. python -u experiments/topology_generalization/run_tp_eval.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --gnn_type gat \
  --topologies TP1 TP2 TP3 TP4 TP5 TP6 TP7 \
  --episodes 31 \
  --include_zero \
  --include_random \
  --run_name "${RUN_NAME}"

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit ${rc}