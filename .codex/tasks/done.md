# Done

- 2026-08-14: Fast-forward merged `sapc-finetune-pipeline` into local `main`; both local branches now point at commit `e71e28a`.
- 2026-08-14: Cross-checked `sapc-finetune-pipeline` against `main`; local `main` was an ancestor of the branch, `git merge-tree --write-tree main HEAD` reported no conflicts, and the branch was safe to fast-forward merge into local `main`.
- 2026-08-14: Moved the standalone MeanVC2 fine-tune repository to `/home/evan1923/projects/JC-meanVC2`, changed its `origin` remote to `git@github.com:Evanchan1923/MeanVC2.git`, and created the `sapc-finetune-pipeline` branch for the transferred pipeline work.
- 2026-08-14: Transferred the current SAPC MeanVC2 fine-tune pipeline and `.codex` project settings into `JC-meanVC2`, patched the inner MeanVC2 training entrypoint for pretrained checkpoint loading and artifact checkpoint paths, and moved the outer gitlink tracking from `MeanVC2-JC` to `JC-meanVC2`.
- 2026-08-14: Compared `MeanVC2-JC` against the current MeanVC repo and ported the MeanVC2 model path into the current SAPC fine-tune workflow, preserving the PBS/YAML launcher structure and checkpoint flow.
- 2026-08-13: Added a MeanVC fine-tuning pipeline for the SAPC Severe Hugging Face dataset, including HF bytes-audio feature preparation, config YAML, PBS launcher, trainer fixes, and static checks.
- 2026-08-13: Renamed the MeanVC fine-tune YAML/PBS files to `meanVC_ft_v1` and updated MeanVC checkpoint paths to `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- 2026-08-13: Moved `meanVC_ft_v1.pbs` to the repository root for direct PBS submission.
- 2026-08-13: Updated all speaker-verification checkpoint defaults to use `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint/wavlm_large_finetune.pth`.
- 2026-08-13: Removed obsolete `example.pbs` and `example.yaml` after adding `meanVC_ft_v1` files.
- 2026-08-13: Reviewed speaker-wise fine-tuning feasibility for MeanVC using `configs/speaker_research_all.csv`.
- 2026-08-13: Created `configs/speaker_research_nutts_gt200.csv` by filtering speakers with `n_utts > 200` and sorting by utterance count descending.
