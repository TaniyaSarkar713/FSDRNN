#!/bin/bash
#SBATCH --job-name=sdr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-i36-[1,3,5,7,9,11,13,15]
#SBATCH -A backfill2
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=FAIL

set -euo pipefail

module purge

eval "$(/gpfs/research/software/python/anaconda312/bin/conda shell.bash hook)"
conda activate spd_frechet

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd /gpfs/home/ts22u/FSDRNN/code_taniya
mkdir -p logs

echo "=== host ==="
hostname

echo "=== visible gpu(s) ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

SETUPS=(
    "simulations_sdr/setup1_linear.py"
    "simulations_sdr/setup2_linear_p50.py"
    "simulations_sdr/setup3_linear_v20.py"
    "simulations_sdr/setup4_nonlinear_z.py"
    "simulations_sdr/setup5_correlated_responses.py"
)

NUM_SETUPS=${#SETUPS[@]}
BASE_SEED=42
NUM_REPS=10

SETUP_INDEX=$SLURM_ARRAY_TASK_ID

if [ "$SETUP_INDEX" -ge "$NUM_SETUPS" ]; then
    echo "Invalid SETUP_INDEX=$SETUP_INDEX"
    exit 1
fi

SIM_SCRIPT="${SETUPS[$SETUP_INDEX]}"
SCRIPT_BASENAME=$(basename "$SIM_SCRIPT" .py)

echo "=========================================="
echo "Setup: $SCRIPT_BASENAME"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Base seed: $BASE_SEED"
echo "Number of repetitions: $NUM_REPS"
echo "=========================================="
echo "Running: $SIM_SCRIPT with $NUM_REPS independent runs"
echo ""

# Create results directory and unique output filename
RESULTS_DIR="results_sdr"
mkdir -p "$RESULTS_DIR"

OUTPUT_FILE="${RESULTS_DIR}/${SCRIPT_BASENAME}_seed${BASE_SEED}.json"

python -u "$SIM_SCRIPT" \
    --device cuda \
    --n_reps $NUM_REPS \
    --seed $BASE_SEED \
    --output "$OUTPUT_FILE"