from __future__ import annotations

import argparse
import os
import sys


def parse_cc(value: str) -> tuple[int, int]:
    major, _, minor = value.partition(".")
    return int(major), int(minor or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that the active PyTorch env can run on the allocated Katana GPU.")
    parser.add_argument("--min-cc", default="7.0", help="Minimum CUDA compute capability. V100 is 7.0.")
    parser.add_argument("--allow-no-gpu", action="store_true", help="Print environment info but do not fail when no CUDA GPU is visible.")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"Could not import torch from {sys.executable}: {exc}") from exc

    print(f"[GPU CHECK] Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"[GPU CHECK] Torch: {torch.__version__}")
    print(f"[GPU CHECK] Torch CUDA runtime: {torch.version.cuda}")
    print(f"[GPU CHECK] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[GPU CHECK] TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST', '<unset>')}")

    if not torch.cuda.is_available():
        message = "[GPU CHECK] torch.cuda.is_available() is false; no CUDA GPU is visible."
        if args.allow_no_gpu:
            print(message)
            return
        raise SystemExit(message)

    min_cc = parse_cc(args.min_cc)
    device_count = torch.cuda.device_count()
    print(f"[GPU CHECK] Visible CUDA devices: {device_count}")

    for index in range(device_count):
        name = torch.cuda.get_device_name(index)
        cc = torch.cuda.get_device_capability(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
        print(f"[GPU CHECK] cuda:{index}: {name}, compute capability {cc[0]}.{cc[1]}, memory {total_gb:.1f} GiB")
        if cc < min_cc:
            raise SystemExit(
                f"GPU cuda:{index} has compute capability {cc[0]}.{cc[1]}, below required {args.min_cc}. "
                "This repo targets V100 or newer."
            )


if __name__ == "__main__":
    main()
