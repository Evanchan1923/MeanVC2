# Decisions

- Use `/home/evan1923/projects/JC-meanVC2` as the standalone active MeanVC2 repo path and keep the SAPC fine-tune pipeline files there rather than maintaining them only in the outer MeanVC-based repo.
- For the SAPC fine-tune path, use the MeanVC2 120ms+40ms model (`config_120ms_40ms.json` and `meanvc2_120ms_40ms.safetensors`) as the quality-oriented MeanVC2 option while preserving the current PBS/YAML/HF fine-tune wrapper.
- MeanVC2 fine-tuning should use 160ms BN extraction via `fastu2pp_160ms.pt`; prepared artifacts use `prepared_train_meanvc2_160ms` and run under `meanVC2_ft_v1` to avoid reusing old MeanVC1 features or checkpoints.
- Prompt mel features are obsolete for the active MeanVC2 fine-tune trainer; the trainer consumes `bn`, `mel`, `xvector`, and `inputs_length`.
- The MeanVC SAPC fine-tuning pipeline prepares the Hugging Face dataset into the existing MeanVC manifest format instead of replacing `DiffusionDataset` and `Trainer`.
- HF audio is read by casting the `Audio` column to `decode=False` and decoding the stored bytes with `soundfile`; `torchaudio` and `torchcodec` are not used for audio I/O.
- Validation is opt-in through `run_validation` because the existing trainer validation uses hardcoded paths and the current task focuses on fine-tuning only.
- The fine-tune config and PBS launcher are named `meanVC_ft_v1` for the first server fine-tuning version.
- Full per-speaker MeanVC fine-tuning should not be the default for all SAPC speakers; prefer one shared multi-speaker fine-tune first, then optional per-speaker runs only for selected speakers with enough utterances.
