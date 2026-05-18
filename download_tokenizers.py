"""
Download 4M tokenizer checkpoints from HuggingFace Hub.
Only downloads RGB and depth tokenizers needed for this project.

Usage:
    python download_tokenizers.py
    python download_tokenizers.py --output_dir /path/to/tokenizer_ckpts
    python download_tokenizers.py --token hf_yourtoken  # if models are gated
"""

import argparse
import os
from huggingface_hub import snapshot_download

TOKENIZERS = {
    "rgb": "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
    "depth": "EPFL-VILAB/4M_tokenizers_depth_8k_224-448",
}

def download_tokenizers(output_dir: str, token: str = None):
    os.makedirs(output_dir, exist_ok=True)

    for name, repo_id in TOKENIZERS.items():
        local_dir = os.path.join(output_dir, repo_id.split("/")[-1])
        print(f"\nDownloading {name} tokenizer from {repo_id} -> {local_dir}")

        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
        )
        print(f"  Done: {name}")

    print(f"\nAll tokenizers downloaded to: {output_dir}")
    print("\nCheckpoint paths to use in save_vq_tokens.py:")
    for name, repo_id in TOKENIZERS.items():
        local_dir = os.path.join(output_dir, repo_id.split("/")[-1])
        print(f"  --tokenizer_id {repo_id.split('/')[-1]}  (in {local_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 4M tokenizer checkpoints")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer_ckpts",
        help="Directory to save tokenizer checkpoints (default: ./tokenizer_ckpts)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token if needed (default: None)",
    )
    args = parser.parse_args()
    download_tokenizers(args.output_dir, args.token)