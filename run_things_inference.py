#!/usr/bin/env python3
"""THINGS RGB→Depth inference: load a trained checkpoint, run generation, visualize.

Usage (on Modal where /project paths exist):
  python run_things_inference.py \\
    --checkpoint /project/output/pixel_only/2layer/dim128/checkpoints/checkpoint-latest.pth \\
    --config /opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-2e-2d-scaling.yaml \\
    --things_root /project/data/val/things \\
    --shard_idx 0 --n_samples 4 \\
    --output /project/output/inference/rgb_depth.png

Outputs a PNG with N rows × 3 columns: decoded RGB | GT depth | predicted depth.
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import tarfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Path setup — must happen before importing fourm or neural lib code.
_SCRIPT_DIR = Path(__file__).resolve().parent          # ml-4m/
_REPO_ROOT  = _SCRIPT_DIR.parent                       # neural-foundation-model/
sys.path.insert(0, str(_SCRIPT_DIR))                   # ml-4m  (fourm package)
sys.path.insert(0, str(_REPO_ROOT / "4m_training" / "lib"))  # neural lib

import fourm_neural_modalities  # registers tok_meg_* into MODALITY_INFO — must precede FM imports
import fourm.models.fm           # triggers @register_model decorators for fm_neural_* variants

from neural_trial_transform import MegTrialSampleTransform, is_placeholder
from neural_constants import MEG_TRIAL_SHAPE

from fourm.data.modality_info import MODALITY_INFO
from fourm.models.generate import (
    GenerationSampler,
    build_chained_generation_schedules,
    init_empty_target_modality,
    init_full_input_modality,
)
from fourm.utils import create_model
from fourm.vq.vqvae import DiVAE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_modality_tar(path: Path) -> dict[str, np.ndarray]:
    """Return {key: array} for every .npy file in a single-modality THINGS shard."""
    samples: dict[str, np.ndarray] = {}
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".npy"):
                continue
            # Key is everything before the first dot: "000042" from "000042.tok_rgb.npy"
            key = member.name.split(".")[0]
            buf = tf.extractfile(member).read()
            samples[key] = np.load(io.BytesIO(buf))
    return samples


_MEG_TRANSFORM = MegTrialSampleTransform(training=False)  # always picks trial 0


def load_things_samples(
    things_root: Path,
    shard_idx: int,
    n_samples: int,
    seed: int = 42,
    sample_keys: list[str] | None = None,
    meg_source: str | None = None,
) -> list[dict[str, Any]]:
    """Load tok_rgb and tok_depth numpy arrays from THINGS shard tars.

    THINGS shards are stored one modality per tar:
      things_root/tok_rgb/shard_NNN.tar
      things_root/tok_depth/shard_NNN.tar
    Each contains flat (196,) int16 arrays (no augmentation axis).

    If sample_keys is given those specific keys are loaded (for reproducible
    cross-model comparison); otherwise n_samples random keys are chosen.

    If meg_source is set (e.g. 'tok_meg' or 'tok_meg_avg'), MEG tokens are also
    loaded from the corresponding shard and split into per-RVQ-layer (128,) arrays.
    Samples whose MEG array is a sentinel placeholder are skipped.
    """
    rgb_tar   = things_root / "tok_rgb"   / f"shard_{shard_idx:03d}.tar"
    depth_tar = things_root / "tok_depth" / f"shard_{shard_idx:03d}.tar"

    for p in (rgb_tar, depth_tar):
        if not p.exists():
            raise FileNotFoundError(f"Shard not found: {p}")

    rgb_samples   = _read_modality_tar(rgb_tar)
    depth_samples = _read_modality_tar(depth_tar)

    meg_samples: dict[str, np.ndarray] = {}
    if meg_source:
        meg_tar = things_root / meg_source / f"shard_{shard_idx:03d}.tar"
        if not meg_tar.exists():
            raise FileNotFoundError(f"MEG shard not found: {meg_tar}")
        meg_samples = _read_modality_tar(meg_tar)

    common_keys = sorted(set(rgb_samples) & set(depth_samples))
    if meg_source:
        common_keys = sorted(set(common_keys) & set(meg_samples))
    if not common_keys:
        raise RuntimeError("No common keys across requested modality shards.")

    print(f"  Available keys in shard ({len(common_keys)} total): {common_keys[:8]} ...")

    rng = random.Random(seed)
    if sample_keys is not None:
        missing = [k for k in sample_keys if k not in set(common_keys)]
        if missing:
            print(f"  WARNING: keys not in shard, falling back to seeded random: {missing}")
            selected = rng.sample(common_keys, min(n_samples, len(common_keys)))
        else:
            selected = sample_keys
    else:
        selected = rng.sample(common_keys, min(n_samples, len(common_keys)))

    results = []
    for k in selected:
        entry: dict[str, Any] = {
            "key": k,
            "tok_rgb":   rgb_samples[k].astype(np.int64),
            "tok_depth": depth_samples[k].astype(np.int64),
        }
        if meg_source:
            arr = meg_samples[k]
            if is_placeholder(arr, MEG_TRIAL_SHAPE):
                print(f"  Skipping key {k}: MEG is a sentinel placeholder.")
                continue
            grid, _ = _MEG_TRANSFORM(arr)   # (128, 4) int32
            entry["meg_rvq"] = [np.ascontiguousarray(grid[:, q]) for q in range(4)]  # list of (128,)
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# MEG input helper
# ---------------------------------------------------------------------------

def add_meg_input(
    mod_dict: dict,
    meg_rvq: list[np.ndarray],
    modality_info: dict,
    device: torch.device,
) -> tuple[dict, list[str]]:
    """Add per-RVQ-layer MEG arrays as full encoder inputs.

    Only adds layers that are present in modality_info (handles rvq0-only models).
    Returns the updated mod_dict and the list of added domain names.
    """
    added = []
    for q, arr in enumerate(meg_rvq):
        domain = f"tok_meg_rvq{q}"
        if domain not in modality_info:
            continue
        t = torch.tensor(arr, dtype=torch.int64).unsqueeze(0).to(device)  # (1, 128)
        mod_dict[domain] = {"tensor": t}
        mod_dict = init_full_input_modality(mod_dict, modality_info, domain, device)
        added.append(domain)
    return mod_dict, added


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_fm_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, dict]:
    """Build FM model from a training YAML config and load a checkpoint."""
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)

    # Resolve data config (may use /opt/repo/ prefix on Modal)
    data_cfg_path = Path(model_cfg["data_config"])
    if not data_cfg_path.exists():
        # Try resolving relative to repo root for local use
        alt = _REPO_ROOT / "ml-4m" / data_cfg_path.relative_to("/opt/repo/ml-4m")
        if alt.exists():
            data_cfg_path = alt
        else:
            raise FileNotFoundError(
                f"data_config not found: {model_cfg['data_config']}\n"
                "If running locally, set FOURM_REPO or edit the config path."
            )

    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)

    # Collect all in/out domains across all train datasets
    train_datasets = data_cfg["train"]["datasets"]
    in_domains = sorted(set.union(*[
        set(cfg["in_domains"].split("-")) for cfg in train_datasets.values()
    ]))
    out_domains = sorted(set.union(*[
        set(cfg["out_domains"].split("-")) for cfg in train_datasets.values()
    ]))
    all_domains = sorted(set(in_domains) | set(out_domains))

    # Build modality_info (mirrors run_training_4m.py:setup_modality_info)
    input_size = model_cfg.get("input_size", 224)
    patch_size = model_cfg.get("patch_size", 16)
    modality_info = {mod: dict(MODALITY_INFO[mod]) for mod in all_domains}
    for mod, info in modality_info.items():
        if info["type"] == "img":
            img_sz = info.get("input_size", input_size)
            p_sz   = info.get("patch_size",  patch_size)
            info["max_tokens"] = (img_sz // p_sz) ** 2

    # Build embeddings (mirrors run_training_4m.py:get_model)
    encoder_embeddings: dict = {}
    for mod in in_domains:
        info = modality_info[mod]
        if info.get("encoder_embedding") is None:
            continue
        if info["type"] == "img":
            img_sz = info.get("input_size", input_size)
            p_sz   = info.get("patch_size",  patch_size)
            encoder_embeddings[mod] = info["encoder_embedding"](patch_size=p_sz, image_size=img_sz)
        else:
            encoder_embeddings[mod] = info["encoder_embedding"]()

    decoder_embeddings: dict = {}
    for mod in out_domains:
        info = modality_info[mod]
        if info.get("decoder_embedding") is None:
            continue
        if info["type"] == "img":
            img_sz = info.get("input_size", input_size)
            p_sz   = info.get("patch_size",  patch_size)
            decoder_embeddings[mod] = info["decoder_embedding"](patch_size=p_sz, image_size=img_sz)
        else:
            decoder_embeddings[mod] = info["decoder_embedding"]()

    model = create_model(
        model_cfg["model"],
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        num_register_tokens=model_cfg.get("num_register_tokens", 0),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {model_cfg['model']} ({n_params:.1f}M params) from {checkpoint_path.name}")
    return model, modality_info


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_to_image(
    tokenizer: DiVAE,
    tokens: torch.Tensor,
    is_rgb: bool,
    timesteps: int = 25,
    image_size: int = 224,
) -> np.ndarray:
    """Decode integer tokens (1, 196) → numpy image array for display.

    RGB: returns (224, 224, 3) float32 in [0, 1].
    Depth: returns (224, 224) float32 in [0, 1], min-max normalised.
    timesteps: diffusion steps for DiVAE decoder (default 25, matching 4M demo;
               1000 train-time default hangs for minutes).
    """
    tokens_2d = tokens.reshape(1, 14, 14)
    with torch.no_grad():
        img = tokenizer.decode_tokens(tokens_2d, timesteps=timesteps, image_size=image_size)
    img = img.squeeze(0).cpu().float()
    if is_rgb:
        img = img * 0.5 + 0.5                      # [-1, 1] → [0, 1]
        img = img.permute(1, 2, 0).numpy()         # (224, 224, 3)
        return np.clip(img, 0.0, 1.0)
    else:
        img = img.squeeze(0).numpy()               # (224, 224)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)           # min-max → [0, 1]
        return img.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    import os
    device = torch.device(args.device)

    # Redirect HF cache to a persistent location (e.g. Modal project volume) so
    # tokenizers are downloaded once and reused across runs.
    if args.tokenizer_cache:
        cache_dir = Path(args.tokenizer_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        print(f"HF tokenizer cache → {cache_dir}")

    print("Loading FM model…")
    model, modality_info = build_fm_model(
        Path(args.config), Path(args.checkpoint), device
    )
    sampler = GenerationSampler(model)

    print("Loading tokenizers…")
    tok_rgb_dec   = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_rgb_16k_224-448").to(device).eval()
    tok_depth_dec = DiVAE.from_pretrained("EPFL-VILAB/4M_tokenizers_depth_8k_224-448").to(device).eval()

    sample_keys = [k.strip() for k in args.sample_key.split(",")] if args.sample_key else None
    meg_source = args.meg_source if args.include_meg else None
    n = len(sample_keys) if sample_keys else args.n_samples
    print(f"Loading {n} THINGS samples from shard {args.shard_idx}…")
    samples = load_things_samples(
        Path(args.things_root), args.shard_idx, args.n_samples,
        seed=args.seed, sample_keys=sample_keys, meg_source=meg_source,
    )
    print(f"  Keys: {[s['key'] for s in samples]}")

    n = len(samples)
    input_label = f"RGB+MEG ({args.meg_source})" if args.include_meg else "RGB"
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n), squeeze=False)
    col_titles = ["RGB (decoded)", "Depth GT", f"Depth predicted\n({input_label} → depth)"]

    for row, sample in enumerate(samples):
        tok_rgb_t   = torch.tensor(sample["tok_rgb"],   dtype=torch.int64).unsqueeze(0).to(device)
        tok_depth_t = torch.tensor(sample["tok_depth"], dtype=torch.int64).unsqueeze(0).to(device)

        mod_dict: dict = {"tok_rgb@224": {"tensor": tok_rgb_t}}
        mod_dict = init_full_input_modality(mod_dict, modality_info, "tok_rgb@224", device)

        meg_cond_domains: list[str] = []
        if args.include_meg and "meg_rvq" in sample:
            mod_dict, meg_cond_domains = add_meg_input(
                mod_dict, sample["meg_rvq"], modality_info, device
            )

        mod_dict = init_empty_target_modality(
            mod_dict, modality_info, "tok_depth@224",
            batch_size=1, num_tokens=196, device=device,
        )

        # Build schedule per-sample so cond_domains reflects actual MEG availability
        schedule = build_chained_generation_schedules(
            cond_domains=["tok_rgb@224"] + meg_cond_domains,
            target_domains=["tok_depth@224"],
            tokens_per_target=[196],
            autoregression_schemes=["roar"],
            decoding_steps=[1],
            token_decoding_schedules=["linear"],
            temps=[0.01],
            temp_schedules=["constant"],
            cfg_scales=[2.0],
            cfg_schedules=["constant"],
            cfg_grow_conditioning=False,
            modality_info=modality_info,
        )

        with torch.no_grad():
            mod_dict = sampler.generate(mod_dict, schedule, seed=args.seed)

        pred_tokens = mod_dict["tok_depth@224"]["tensor"]  # (1, 196)

        rgb_img    = decode_to_image(tok_rgb_dec,   tok_rgb_t,   is_rgb=True)
        gt_depth   = decode_to_image(tok_depth_dec, tok_depth_t, is_rgb=False)
        pred_depth = decode_to_image(tok_depth_dec, pred_tokens, is_rgb=False)

        for col, (img, title) in enumerate(zip([rgb_img, gt_depth, pred_depth], col_titles)):
            ax = axes[row][col]
            if col == 0:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="plasma")
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.axis("off")
        axes[row][0].set_ylabel(f"key {sample['key']}", fontsize=8, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="THINGS RGB→Depth inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--config",     required=True, help="Path to model YAML config")
    parser.add_argument("--things_root", required=True,
                        help="Root of THINGS val data, e.g. /project/data/val/things")
    parser.add_argument("--shard_idx",  type=int, default=0, help="Which shard to sample from")
    parser.add_argument("--n_samples",  type=int, default=4, help="Number of images to evaluate")
    parser.add_argument("--include_meg", action="store_true",
                        help="Add MEG tokens as additional encoder input (requires MEG-trained checkpoint).")
    parser.add_argument("--meg_source",  default="tok_meg_avg",
                        help="MEG shard folder to load from (default: tok_meg_avg; use tok_meg for single-trial).")
    parser.add_argument("--sample_key",  default=None,
                        help="Comma-separated shard key(s) to load, e.g. '000042,000117'. "
                             "Overrides --n_samples for reproducible cross-model comparison.")
    parser.add_argument("--tokenizer_cache", default=None,
                        help="Directory to cache HF tokenizers (set to /project/hf_cache on Modal "
                             "so they download once and persist across runs).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     default="inference_out.png", help="Output PNG path")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
