# Project Context

- The active standalone repository path is `/home/evan1923/projects/JC-meanVC2`; its GitHub remote is `git@github.com:Evanchan1923/MeanVC2.git`.
- `JC-meanVC2/` is the active pulled MeanVC2 repository path for ongoing SAPC fine-tuning work; the old outer gitlink path was `MeanVC2-JC`.
- MeanVC downloaded checkpoints are expected under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- MeanVC2 fine-tuning uses the same checkpoint root, with `meanvc2_120ms_40ms.safetensors` and `fastu2pp_160ms.pt` expected under `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- The MeanVC2 SAPC fine-tune run name is `meanVC2_ft_v1`, so artifacts are written under `/srv/scratch/speechdata/SAPC_Team/meanvc_runs/meanVC2_ft_v1`.
- SAPC Severe Hugging Face dataset root is `/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_Severe`, with `train` used for fine-tuning and `dev` reserved for later validation.
- `configs/speaker_research_nutts_gt200.csv` is derived from `configs/speaker_research_all.csv` using `n_utts > 200` and descending `n_utts` order.
