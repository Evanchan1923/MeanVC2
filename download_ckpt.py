from huggingface_hub import snapshot_download
from pathlib import Path

DEFAULT_CKPT_DIR = "/srv/scratch/speechdata/SAPC_Team/meanVC_checkpoint"


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

if __name__ == "__main__":
    download_ckpt()
