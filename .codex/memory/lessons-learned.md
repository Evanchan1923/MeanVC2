# Lessons Learned

- When porting the SAPC wrapper into pulled MeanVC2, `defaults.ini` must include wrapper-passed flags such as `run_validation`, and `src/train/train.py` must explicitly honor `save_dir` plus `pretrained_ckpt_path`.
- MeanVC training CLI flags come from `defaults.ini`; model-required options such as `block_size` must be added there before YAML/PBS launchers can pass `--block-size`.
- MeanVC2 ASR JIT checkpoints are invoked directly with initialized caches, while old MeanVC `fastu2++.pt` exposes `forward_encoder_chunk`; the HF feature preparer needs to support both interfaces when switching checkpoint families.
- `DiffusionDataset.init_data()` returns a list and should be passed as a single constructor argument; expanding it with `*` breaks trainer dataset initialization.
