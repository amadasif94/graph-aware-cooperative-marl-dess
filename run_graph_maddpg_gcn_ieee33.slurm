#!/bin/bash
#SBATCH --job-name=graph_maddpg_gcn_ieee33
#SBATCH --partition=dgx
#SBATCH --qos=freetier
#SBATCH --array=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/graph_maddpg_gcn_%A_%a.out
#SBATCH --error=/network/rit/lab/dahlin_lab/amad/graph_marl_dess/results/logs/slurm/graph_maddpg_gcn_%A_%a.err

set -euo pipefail

BASE_DIR="/network/rit/lab/dahlin_lab/amad/graph_marl_dess"
SLURM_LOG_DIR="${BASE_DIR}/results/logs/slurm"
CHECKPOINT_BASE_DIR="${BASE_DIR}/results/checkpoints/maddpg_ieee33/gcn"
TMP_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/tmp"
PIP_CACHE_DIR_BASE="/network/rit/dgx/dgx_dahlin_lab/amad/graph_marl_dess/pip_cache"

mkdir -p "${SLURM_LOG_DIR}" "${CHECKPOINT_BASE_DIR}" "${TMP_BASE}" "${PIP_CACHE_DIR_BASE}"

export TMPDIR="${TMP_BASE}"
export TEMP="${TMP_BASE}"
export TMP="${TMP_BASE}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR_BASE}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_ID="${SLURM_ARRAY_TASK_ID}"
SEED=$((42 + RUN_ID))
RUN_NAME="gcn_run_${RUN_ID}"
RUN_CHECKPOINT_DIR="${CHECKPOINT_BASE_DIR}/${RUN_NAME}"

mkdir -p "${RUN_CHECKPOINT_DIR}"

echo "============================================================"
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID}"
echo "RUN_ID               = ${RUN_ID}"
echo "RUN_NAME             = ${RUN_NAME}"
echo "SEED                 = ${SEED}"
echo "GNN_TYPE             = gcn"
echo "RUN_CHECKPOINT_DIR   = ${RUN_CHECKPOINT_DIR}"
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
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

from environments.dess_env import DESSEnv
from models.maddpg import GraphAwareMADDPG
from training.trainer import GraphMADDPGTrainer
print("GCN import test passed")
PY

echo "============================================================"
echo "Launching Graph-Aware MADDPG-GCN training for ${RUN_NAME}"
echo "============================================================"

PYTHONPATH=. python -u scripts/run_train_ieee33.py \
    --seed "${SEED}" \
    --gnn_type gcn \
    --total_steps 200000 \
    --warmup_steps 10000 \
    --buffer_size 10000 \
    --batch_size 128 \
    --actor_lr 5e-5 \
    --critic_lr 5e-4 \
    --noise_std 0.05 \
    --eval_every 5000 \
    --save_every 5000 \
    --eval_episodes 10 \
    --checkpoint_dir "${RUN_CHECKPOINT_DIR}"

rc=$?

echo "============================================================"
echo "END TIME      = $(date)"
echo "PYTHON EXIT   = ${rc}"
echo "============================================================"

exit ${rc}