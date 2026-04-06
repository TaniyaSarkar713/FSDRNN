#!/bin/bash
#SBATCH --job-name=sdr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-i36-[1,3,5,7,9,11,13,15]  # exclude GTX 1080 Ti nodes (sm_61)
#SBATCH -A backfill2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=FAIL

set -euo pipefail

module purge
# module load cuda   # only if your cluster requires it

eval "$(/gpfs/research/software/python/anaconda312/bin/conda shell.bash hook)"
conda activate spd_frechet

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /gpfs/home/ts22u/FSDRNN/code_taniya
mkdir -p logs

echo "=== nvidia-smi ==="
nvidia-smi -L || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python simulations/test_sdr_spd.py --device cuda