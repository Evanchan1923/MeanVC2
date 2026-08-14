# MeanVC2 Upstream Fine-Tuning Audit

Date: 2026-08-14

Upstream checked: `https://github.com/ASLP-lab/MeanVC2`, `main` commit `0d39c8a`.

## Upstream Fine-Tuning Contract

The upstream README says MeanVC2 supports speaker-specific fine-tuning from a pretrained safetensors checkpoint. Its recommended quality path is the `120ms+40ms` model. Training data is expected as a `.list` file where each line is:

```text
utt_id|/path/to/bn.npy|/path/to/mel.npy|/path/to/xvector.npy
```

The upstream preparation path is:

1. Extract 10 ms mel spectrograms.
2. Extract BN features with `fastu2pp_160ms.pt` for the `120ms+40ms` model.
3. Extract speaker embeddings.
4. Create the training file list.
5. Run `scripts/train_120ms_40ms.sh` for the quality-oriented model.

## Local Alignment

The SAPC pipeline keeps the upstream model/trainer contract:

- Uses `src/config/config_120ms_40ms.json`.
- Initializes from `meanvc2_120ms_40ms.safetensors`.
- Uses `fastu2pp_160ms.pt` for BN extraction.
- Trains with `feature_list = bn mel xvector` and `additional_feature_list = inputs_length`.
- Produces the upstream four-field manifest format.
- Uses `chunk_size = 12` and `block_size = 4`, matching upstream `120ms+40ms` training.

## Intentional Local Changes

- Added SAPC Hugging Face dataset preparation in `src/train/finetune_meanvc_hf.py`.
- Added `configs/meanVC_ft_v1.yaml` and `meanVC_ft_v1.pbs` for the UNSW/PBS scratch layout.
- Added shared checkpoint download helpers that place MeanVC2 files under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- Added `save_dir` and `pretrained_ckpt_path` handling in `src/train/train.py`; this makes the README's safetensors initialization path explicit.
- Gated hardcoded validation behind `run_validation = 0` for fine-tuning runs without upstream `val/` assets.
- Removed obsolete prompt-mel wiring from active fine-tuning inputs.

## Remaining Differences To Know

The SAPC YAML uses a smaller pilot fine-tuning schedule than upstream's shell script. Upstream `scripts/train_120ms_40ms.sh` uses batch size 16, `epochs=1000`, `learning_rate=1e-4`, `num_warmup_updates=20000`, `save_per_updates=10000`, `flow_ratio=0.75`, and `cfg_ratio=0.2`. The SAPC config currently uses batch size 8, `epochs=20`, `learning_rate=5e-5`, `num_warmup_updates=500`, `save_per_updates=1000`, `flow_ratio=0.5`, and `cfg_ratio=0.1`.

Those schedule changes are dataset/run-size choices, not structural incompatibilities. For a strict upstream-style full fine-tune, adjust the SAPC YAML to match the upstream script values.

## Audit Fixes Applied

- Changed the SAPC manifest writer to emit exactly the upstream four fields.
- Changed HF BN extraction to use full upstream-sized BN windows instead of a short trailing partial window.
- Truncated generated mel frames to the upstream hop-boundary frame count.
