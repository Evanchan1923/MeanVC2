from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"[OK] {label}: {path} ({path.stat().st_size / (1024 ** 2):.1f} MiB)")


def check_imports() -> None:
    modules = [
        "accelerate",
        "datasets",
        "einops",
        "librosa",
        "safetensors",
        "soundfile",
        "torchaudio",
        "torchdiffeq",
        "wandb",
        "yaml",
    ]
    missing = []
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing.append(f"{module}: {exc}")
        else:
            print(f"[OK] import {module}")
    if missing:
        raise RuntimeError("Missing or broken Python dependencies:\n  " + "\n  ".join(missing))


def resolve_debug_context(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path, dict[str, Any]]:
    from src.train.finetune_meanvc_hf import load_yaml_config, to_path, train_args_from_config

    cfg = load_yaml_config(config_path)
    repo_root = to_path(cfg["user_settings"].get("repo_root", "."), Path.cwd()).resolve()
    run_root = to_path(cfg["run"]["run_root"]).resolve()
    run_name = str(cfg["run"]["name"])
    run_dir = run_root / run_name
    output_dir = run_dir / cfg["run"].get("output_subdir", "artifacts")
    manifest_path = output_dir / cfg["prepare"].get("output_subdir", "prepared_train") / "train_manifest.txt"
    train_args = train_args_from_config(cfg["train"], manifest_path, run_dir, output_dir)
    return cfg, repo_root, output_dir, manifest_path, train_args


def check_config_paths(cfg: dict[str, Any], repo_root: Path, train_args: dict[str, Any]) -> None:
    from src.train.finetune_meanvc_hf import to_path

    dataset_path = to_path(cfg["user_settings"]["hf_dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"HF dataset path not found: {dataset_path}")
    print(f"[OK] HF dataset path exists: {dataset_path}")

    model_config = to_path(train_args["model_config"], repo_root)
    require_file(model_config, "MeanVC2 model config")

    require_file(Path(train_args["pretrained_ckpt_path"]).expanduser(), "MeanVC2 pretrained safetensors")
    require_file(Path(cfg["prepare"]["asr_ckpt_path"]).expanduser(), "FastU2++ ASR checkpoint")
    require_file(Path(cfg["prepare"]["speaker_verification_ckpt_path"]).expanduser(), "speaker verification checkpoint")

    vocoder_path = Path(cfg["user_settings"]["checkpoint_root"]).expanduser() / "vocos.pt"
    require_file(vocoder_path, "Vocos checkpoint")


def check_model_load(repo_root: Path, train_args: dict[str, Any], device: torch.device) -> None:
    from src.model import DiT
    from src.model.utils import load_checkpoint
    from src.train.finetune_meanvc_hf import to_path

    model_config_path = to_path(train_args["model_config"], repo_root)
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)
    model = DiT(**model_config["model"])
    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"[OK] Built DiT model: {total_params:.3f}M params")

    model = load_checkpoint(model, train_args["pretrained_ckpt_path"], device="cpu", use_ema=True)
    model.to(device)
    model.eval()
    print(f"[OK] Loaded pretrained checkpoint into DiT on {device}")


def check_torchscript(path: Path, label: str, device: torch.device) -> None:
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    print(f"[OK] torch.jit.load {label} on {device}")
    del module


def check_manifest(manifest_path: Path, max_rows: int) -> None:
    if not manifest_path.exists():
        print(f"[WARN] Prepared manifest not found yet: {manifest_path}")
        print("[WARN] This is okay before the first prepare run.")
        return

    rows = [line.strip().split("|") for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    print(f"[OK] Manifest exists with {len(rows)} rows: {manifest_path}")

    for row_index, row in enumerate(rows[:max_rows]):
        if len(row) != 4:
            raise RuntimeError(f"Manifest row {row_index} has {len(row)} fields, expected 4: {'|'.join(row)}")
        utt, bn_path, mel_path, xvector_path = row
        for label, value in (("bn", bn_path), ("mel", mel_path), ("xvector", xvector_path)):
            require_file(Path(value).expanduser(), f"manifest row {row_index} {label}")
        bn = np.load(bn_path, mmap_mode="r")
        mel = np.load(mel_path, mmap_mode="r")
        xvector = np.load(xvector_path, mmap_mode="r")
        print(f"[OK] manifest row {row_index} {utt}: bn={bn.shape}, mel={mel.shape}, xvector={xvector.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug MeanVC2 SAPC fine-tune environment/checkpoints without starting training.")
    parser.add_argument("--config", default="configs/meanVC_ft_v1.yaml")
    parser.add_argument("--skip-model-load", action="store_true", help="Skip loading the pretrained VC model.")
    parser.add_argument("--skip-jit-load", action="store_true", help="Skip torch.jit.load checks for ASR/vocoder.")
    parser.add_argument("--manifest-rows", type=int, default=3, help="Prepared manifest rows to inspect if the manifest exists.")
    args = parser.parse_args()

    print(f"[INFO] Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"[INFO] Torch: {torch.__version__}, torch CUDA runtime: {torch.version.cuda}, cuda available: {torch.cuda.is_available()}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    config_path = Path(args.config).expanduser().resolve()
    require_file(config_path, "fine-tune YAML config")
    check_imports()

    cfg, repo_root, output_dir, manifest_path, train_args = resolve_debug_context(config_path)
    print(f"[INFO] repo_root={repo_root}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] manifest_path={manifest_path}")
    check_config_paths(cfg, repo_root, train_args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.skip_model_load:
        check_model_load(repo_root, train_args, device)

    if not args.skip_jit_load:
        check_torchscript(Path(cfg["prepare"]["asr_ckpt_path"]).expanduser(), "FastU2++ ASR", device)
        check_torchscript(Path(cfg["user_settings"]["checkpoint_root"]).expanduser() / "vocos.pt", "Vocos", device)

    check_manifest(manifest_path, args.manifest_rows)
    print("[OK] MeanVC2 fine-tune debug checks completed.")


if __name__ == "__main__":
    main()
