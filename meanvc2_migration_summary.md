# MeanVC2 Migration Summary

Date: 2026-08-14

## Decision

`MeanVC2-JC` is the better model target for the current fine-tune work. It is a newer MeanVC2 implementation focused on lower-latency and more robust zero-shot conversion. The important model changes are Future-Receptive Chunking and a Universal Timbre Token style encoder that uses global speaker embeddings instead of relying on prompt mel audio.

I did not run an audio-quality evaluation locally, so this decision is based on the code and MeanVC2 project documentation, not on SAPC output listening tests.

## Key Differences

| Area | Current MeanVC before migration | `MeanVC2-JC` |
| --- | --- | --- |
| Timbre conditioning | MRTE with prompt mel references | Global timbre memory plus temporal timbre encoder from speaker embedding |
| Model input contract | `bn`, `mel`, `xvector`, `prompt` | `bn`, `mel`, `xvector`; no prompt mel needed |
| Attention | Chunk attention with old cache/prompt flow | Block future-receptive attention with `chunk_size` and `block_size` |
| Main quality config | `config_200ms.json` / `model_200ms.safetensors` | `config_120ms_40ms.json` / `meanvc2_120ms_40ms.safetensors` |
| Fine-tune plumbing | Custom SAPC HF preparer, YAML, PBS job | Not present in `MeanVC2-JC`; preserved from this repo |

## Repo Updates Made

- Ported MeanVC2 model code into:
  - `src/model/backbones/dit.py`
  - `src/model/modules.py`
  - `src/model/cfm_mean_flow.py`
- Added MeanVC2 dataset/config files:
  - `src/dataset/npy_dataset.py`
  - `src/config/config_120ms_40ms.json`
  - `src/config/config_40ms_40ms.json`
- Kept the current fine-tune launcher and checkpoint behavior in `src/train/train.py`.
- Updated `src/model/trainer.py` so training no longer loads or passes prompt mel features.
- Updated `configs/meanVC_ft_v1.yaml` to use the MeanVC2 120ms+40ms model, 160ms BN extraction, and a MeanVC2-specific run name.
- Updated `download_ckpt.py` to download MeanVC2 checkpoint files into the same checkpoint root used by the current repo.

The PBS entry point is still:

```bash
qsub meanVC_ft_v1.pbs
```

The run name inside the YAML is now `meanVC2_ft_v1`, so new artifacts will go under:

```bash
/srv/scratch/speechdata/SAPC_Team/meanvc_runs/meanVC2_ft_v1
```

This avoids accidentally resuming old MeanVC1 checkpoints or reusing old prepared BN features.

## Checkpoints

Checkpoint root stays the same:

```bash
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint
```

Required for the updated fine-tune job:

```text
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/meanvc2_120ms_40ms.safetensors
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/fastu2pp_160ms.pt
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/wavlm_large_finetune.pth
```

Optional but downloaded by `download_ckpt.py`:

```text
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/meanvc2_40ms_40ms.safetensors
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/fastu2pp_80ms.pt
/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/vocos.pt
```

Run this on the server to download the Hugging Face checkpoint files to the same path:

```bash
python initialization.py --task all
```

`wavlm_large_finetune.pth` is still manual. If it is already present in the checkpoint root from the current MeanVC setup, no action is needed for that file.

## Server Venv

If the current `sapc2` conda env already runs the existing fine-tune job with `torch==2.5.1` and CUDA 12.1, you can keep using it. Install the updated requirements in that env:

```bash
conda activate sapc2
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Important packages added or pinned for this migration include:

```text
accelerate==1.7.0
einops==0.8.0
ema-pytorch==0.7.7
x-transformers==2.2.11
safetensors
huggingface-hub>=0.25
soxr>=0.5
prefigure==0.0.10
wandb==0.23.1
```

MeanVC2 upstream recommends Python 3.11. The current PBS script still activates `sapc2` after loading `python/3.10.8`; no PBS change is required unless dependency installation fails in `sapc2`. If you create a new env, submit with:

```bash
CONDA_ENV=meanvc2 qsub meanVC_ft_v1.pbs
```
