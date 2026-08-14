# Katana GPU Portability

This repo targets Katana V100, A100, H100, and H200 nodes with one Python environment. V100 is the minimum compatibility target.

## Policy

- Use Python 3.11 from conda, matching upstream MeanVC2.
- Use PyTorch 2.5.1 with CUDA 12.1 wheels.
- Load CUDA and ffmpeg modules for system libraries.
- Do not load a `python/...` module for training jobs.
- Keep Accelerate mixed precision set to `no`; BF16 is not portable to V100.
- Build any source CUDA extensions with `TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0`.

The architecture list covers V100 (`sm70`), A100 (`sm80`), and H100/H200 (`sm90`). If packages do not build CUDA extensions, the setting is harmless; if they do, the environment remains usable across these GPU classes.

Reference points:

- PyTorch lists official 2.5.1 CUDA 12.1 install commands at <https://pytorch.org/get-started/previous-versions/>.
- NVIDIA lists A100 as compute capability 8.0 and H100/H200 as 9.0 at <https://developer.nvidia.com/cuda/gpus>.
- NVIDIA lists V100 as compute capability 7.0 at <https://developer.nvidia.com/cuda/gpus/legacy>.

## One-Time Environment Setup

From the repo root on Katana:

```bash
bash scripts/setup_katana_meanvc2_env.sh
```

Defaults:

```text
CONDA_ROOT=/srv/scratch/z5327748/miniforge3
CONDA_ENV=/srv/scratch/z5327748/conda_envs/meanvc2
CUDA_MODULE=cuda/12.1.1
FFMPEG_MODULE=ffmpeg/7.0.2
TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0
```

Override them only when needed:

```bash
CONDA_ENV=/srv/scratch/z5327748/conda_envs/meanvc2_py311 bash scripts/setup_katana_meanvc2_env.sh
```

## Job Submission

Run the debug job first to validate the conda environment, GPU, checkpoints, config, and optional prepared manifest:

```bash
qsub meanVC_ft_debug.pbs
```

The PBS script is hardware-model agnostic:

```bash
qsub meanVC_ft_v1.pbs
```

It purges inherited modules, loads CUDA/ffmpeg, activates the conda env, verifies Python 3.11, then checks the visible GPU compute capability before training.

## Manual Sanity Check

Inside an interactive GPU job:

```bash
module purge
module load ffmpeg/7.0.2
module load cuda/12.1.1
source /srv/scratch/z5327748/miniforge3/etc/profile.d/conda.sh
conda activate /srv/scratch/z5327748/conda_envs/meanvc2
python scripts/check_katana_gpu.py --min-cc 7.0
```

Expected result: Python is 3.11, Torch reports CUDA 12.1, and every visible GPU has compute capability at least 7.0.
