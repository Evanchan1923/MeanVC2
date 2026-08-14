#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/srv/scratch/z5327748/miniforge3}"
CONDA_ENV="${CONDA_ENV:-/srv/scratch/z5327748/conda_envs/meanvc2}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"
FFMPEG_MODULE="${FFMPEG_MODULE:-ffmpeg/7.0.2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

# Build any local CUDA extensions for all Katana GPU classes we target:
# V100=sm70, A100=sm80, H100/H200=sm90.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

module purge
module load "${FFMPEG_MODULE}"
module load "${CUDA_MODULE}"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "[ERROR] Could not find conda. Set CONDA_ROOT to your miniforge/miniconda root." >&2
  exit 1
fi

if [[ ! -d "${CONDA_ENV}" ]]; then
  conda create -y -p "${CONDA_ENV}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV}"

echo "[INFO] Conda env: ${CONDA_PREFIX}"
echo "[INFO] Python: $(python --version) ($(command -v python))"
python - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version.split()[0]} at {sys.executable}")
PY

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url "${PYTORCH_INDEX_URL}"
python -m pip install -r "${REPO_ROOT}/requirements.txt"

python "${REPO_ROOT}/scripts/check_katana_gpu.py" --allow-no-gpu

cat <<EOF
[INFO] MeanVC2 environment is ready.
[INFO] Submit with:
       CONDA_ENV=${CONDA_ENV} qsub meanVC_ft_v1.pbs
EOF
