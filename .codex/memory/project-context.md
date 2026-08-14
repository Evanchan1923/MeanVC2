# Project Context

- The active standalone repository path is `/home/evan1923/projects/JC-meanVC2`; its GitHub remote is `git@github.com:Evanchan1923/MeanVC2.git`.
- Upstream ASLP-lab/MeanVC2 `main` was checked at commit `0d39c8a` on 2026-08-14; the comparison is documented in `meanvc2_upstream_finetune_audit.md`.
- As of 2026-08-14, local `main` has been fast-forward merged to `sapc-finetune-pipeline` at commit `e71e28a`; `origin/main` remains behind until `main` is pushed.
- The previous nested MeanVC2 paths under `/home/evan1923/projects/MeanVC2_JC` were `JC-meanVC2/` and the older gitlink `MeanVC2-JC`; ongoing work should use this standalone repository.
- MeanVC downloaded checkpoints are expected under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- MeanVC2 fine-tuning uses the same checkpoint root, with `meanvc2_120ms_40ms.safetensors` and `fastu2pp_160ms.pt` expected under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- MeanVC2 speaker embedding needs both repo-local `preprocess/ckpts/wavlm_large.pt` and shared `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/wavlm_large_finetune.pth`; the fine-tuned speaker-verification checkpoint comes from upstream's Google Drive link, not the MeanVC2 Hugging Face snapshot.
- The SAPC MeanVC2 manifest should follow upstream's four-field format: `utt_id|bn.npy|mel.npy|xvector.npy`.
- Katana MeanVC2 jobs should activate `/srv/scratch/z5327748/conda_envs/meanvc2` from `/srv/scratch/z5327748/miniforge3`; the PBS script checks that active Python is 3.11.
- Katana GPU portability targets V100/A100/H100/H200 with V100 as the floor; use `TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0` when installing packages that may compile CUDA extensions.
- Use `qsub meanVC_ft_debug.pbs` before full training to validate the Katana env, GPU, checkpoints, config, and any prepared manifest.
- The MeanVC2 SAPC fine-tune run name is `meanVC2_ft_v1`, so artifacts are written under `/srv/scratch/speechdata/SAPC_Team/meanvc_runs/meanVC2_ft_v1`.
- SAPC Severe Hugging Face dataset root is `/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_Severe`, with `train` used for fine-tuning and `dev` reserved for later validation.
- `configs/speaker_research_nutts_gt200.csv` is derived from `configs/speaker_research_all.csv` using `n_utts > 200` and descending `n_utts` order.
