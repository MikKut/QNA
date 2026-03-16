#!/bin/bash
#SBATCH --job-name=qna_scout
#SBATCH --partition=tiger
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-14
#SBATCH --output=/dcs/pg24/u5744690/QNA/logs/qna_scout_%A_%a.out
#SBATCH --error=/dcs/pg24/u5744690/QNA/logs/qna_scout_%A_%a.err

set -euo pipefail

REPO_ROOT="/dcs/pg24/u5744690/QNA"
CONFIG_DIR="$REPO_ROOT/configs/generated/scouting_v1"

cd "$REPO_ROOT"

source /etc/profile.d/modules.sh

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found in $REPO_ROOT"
  exit 1
fi

source .venv/bin/activate

if [ ! -d "$CONFIG_DIR" ]; then
  echo "ERROR: config dir not found: $CONFIG_DIR"
  exit 1
fi

mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)

if [ "${#CONFIGS[@]}" -ne 15 ]; then
  echo "ERROR: expected 15 configs in $CONFIG_DIR, found ${#CONFIGS[@]}"
  printf '%s\n' "${CONFIGS[@]}"
  exit 1
fi

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "====================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host:          $(hostname)"
echo "Start:         $(date)"
echo "Config:        $CFG"
echo "====================================="

python -m Code.trainers.train_quantum --config "$CFG"

echo "Finished: $(date)"