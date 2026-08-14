# Lessons Learned

- Upstream MeanVC2 README recommends the 120ms+40ms path for quality and a four-field `.list` manifest; SAPC-specific preparation should produce that same contract rather than carrying old prompt-mel fields.
- When `main` is an ancestor of `sapc-finetune-pipeline`, `git merge --ff-only sapc-finetune-pipeline` cleanly updates local `main` without creating a merge commit.
- `git merge-tree --write-tree main HEAD` can verify the `sapc-finetune-pipeline` merge result without switching branches or modifying the working tree.
- When porting the SAPC wrapper into pulled MeanVC2, `defaults.ini` must include wrapper-passed flags such as `run_validation`, and `src/train/train.py` must explicitly honor `save_dir` plus `pretrained_ckpt_path`.
- MeanVC training CLI flags come from `defaults.ini`; model-required options such as `block_size` must be added there before YAML/PBS launchers can pass `--block-size`.
- MeanVC2 ASR JIT checkpoints are invoked directly with initialized caches, while old MeanVC `fastu2++.pt` exposes `forward_encoder_chunk`; the HF feature preparer needs to support both interfaces when switching checkpoint families.
- `DiffusionDataset.init_data()` returns a list and should be passed as a single constructor argument; expanding it with `*` breaks trainer dataset initialization.
