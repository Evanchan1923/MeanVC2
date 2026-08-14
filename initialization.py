from __future__ import annotations

import argparse

from download_ckpt import DEFAULT_CKPT_DIR, download_ckpt


TASKS = {"preprocess", "train_120ms", "train_40ms", "all"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MeanVC2 checkpoints into the shared checkpoint directory.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    args = parser.parse_args()

    download_ckpt(args.ckpt_dir)


if __name__ == "__main__":
    main()
