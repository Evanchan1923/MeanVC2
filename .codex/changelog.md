# Changelog

## 2026-08-14

- Transferred the SAPC MeanVC2 fine-tune pipeline, `.codex` settings, and repo guidance into `JC-meanVC2`, and updated tracking to use that folder as the active MeanVC2 repo path.
- Compared the current MeanVC repo against `MeanVC2-JC` and documented the migration in `meanvc2_migration_summary.md`.
- Ported the MeanVC2 DiT, MeanFlow, block attention, dataset, and 120ms/40ms model configs into the current repo.
- Updated the SAPC fine-tune config to use MeanVC2 120ms+40ms checkpoints, 160ms BN extraction, no prompt mel training feature, and a MeanVC2-specific run artifact path.
- Updated checkpoint download defaults to fetch MeanVC2 files into `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- Added MeanVC2-related Python dependency pins and `block_size` CLI defaults.

## 2026-08-13

- Added MeanVC SAPC Severe fine-tuning pipeline files and training defaults.
- Added GPU Accelerate and PBS launcher configs for server fine-tuning.
- Fixed MeanVC trainer manifest loading and made hardcoded validation opt-in.
- Renamed the fine-tune config and PBS launcher to `meanVC_ft_v1` and updated MeanVC checkpoint paths to `/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint`.
- Moved `meanVC_ft_v1.pbs` to the repository root.
- Updated speaker-verification checkpoint defaults to use the shared `meanVC_checkpoint` directory.
- Removed obsolete `example.pbs` and `example.yaml`.
- Documented the recommendation to keep shared multi-speaker MeanVC fine-tuning as the default and use per-speaker runs selectively.
- Added filtered speaker research CSV for speakers with more than 200 utterances, sorted by utterance count descending.

## 2026-06-30
