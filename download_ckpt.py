from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download

DEFAULT_CKPT_DIR = "/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint"
WAVLM_LARGE_URL = "https://github.com/microsoft/unilm/releases/download/wavlm/WavLM-Large.pt"
WAVLM_FINETUNED_URL = "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"
WAVLM_BASE_PATH = Path("preprocess/ckpts/wavlm_large.pt")
WAVLM_CFG_PATH = Path("preprocess/ckpts/wavlm_large_cfg.pt")
WAVLM_FINETUNED_NAME = "wavlm_large_finetune.pth"


def _print_existing(path: Path, label: str) -> None:
    print(f"[skip] {label}: {path} ({path.stat().st_size / (1024 ** 2):.1f} MiB)")


def _ensure_wavlm_base() -> None:
    WAVLM_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WAVLM_BASE_PATH.is_file():
        _print_existing(WAVLM_BASE_PATH, "WavLM-Large base checkpoint")
    else:
        print(f"[download] WavLM-Large base checkpoint -> {WAVLM_BASE_PATH}")
        urlretrieve(WAVLM_LARGE_URL, WAVLM_BASE_PATH)

    if WAVLM_CFG_PATH.is_file():
        _print_existing(WAVLM_CFG_PATH, "WavLM-Large config")
        return

    try:
        import torch

        checkpoint = torch.load(WAVLM_BASE_PATH, map_location="cpu")
        cfg = checkpoint.get("cfg", checkpoint.get("config"))
        if cfg is None:
            print(f"[warn] Could not extract WavLM config from {WAVLM_BASE_PATH}; runtime streaming may need {WAVLM_CFG_PATH}.")
            return
        torch.save(cfg, WAVLM_CFG_PATH)
        print(f"[done] WavLM-Large config: {WAVLM_CFG_PATH}")
    except Exception as exc:
        print(f"[warn] Could not extract WavLM config from {WAVLM_BASE_PATH}: {exc}")


def _check_manual_wavlm_finetune(dest_dir: str) -> None:
    finetuned_path = Path(dest_dir).expanduser() / WAVLM_FINETUNED_NAME
    if finetuned_path.is_file():
        _print_existing(finetuned_path, "speaker verification checkpoint")
        return

    print("[manual] Speaker verification checkpoint is not included in the ASLP-lab/MeanVC2 Hugging Face files.")
    print(f"[manual] Download wavlm_large_finetune.pth from: {WAVLM_FINETUNED_URL}")
    print(f"[manual] Expected for this SAPC fine-tune config: {finetuned_path}")


def download_ckpt(dest_dir: str = DEFAULT_CKPT_DIR) -> None:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="ASLP-lab/MeanVC2",
        allow_patterns=[
            "meanvc2_120ms_40ms.safetensors",
            "meanvc2_40ms_40ms.safetensors",
            "fastu2pp_160ms.pt",
            "fastu2pp_80ms.pt",
            "vocos.pt",
        ],
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    _ensure_wavlm_base()
    _check_manual_wavlm_finetune(dest_dir)

if __name__ == "__main__":
    download_ckpt()
