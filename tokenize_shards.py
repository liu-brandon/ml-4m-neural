"""
Tokenizes WebDataset shards for RGB and depth modalities, writing output
token shards and crop settings shards in the same WebDataset format.

Output structure:
    root/tok_rgb@224/shard-00000.tar
    root/tok_depth@224/shard-00000.tar
    root/crop_settings/shard-00000.tar

Usage:
    # Dry run first
    python tokenize_shards.py --data_root /path/to/data --task rgb --dryrun
    python tokenize_shards.py --data_root /path/to/data --task depth --dryrun

    # Real runs (rgb first to generate crop settings, then depth)
    python tokenize_shards.py --data_root /path/to/data --task rgb --n_crops 4
    python tokenize_shards.py --data_root /path/to/data --task depth --n_crops 4
"""

import argparse
import datetime
import hashlib
import io
import os
import glob
import time
import math

import numpy as np
import torch
import webdataset as wds
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from fourm.data import CenterCropImageAugmenter, RandomCropImageAugmenter
from fourm.data.modality_info import MODALITY_TRANSFORMS_DIVAE
from fourm.vq.vqvae import DiVAE

from torch.utils.data import DataLoader, IterableDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASK_CONFIG = {
    'rgb': {
        'hf_id':        'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448',
        'wds_key':      'jpg',          # key inside the raw tar
        'tok_folder':   'tok_rgb@224',  # output folder name
        'resample_mode':'bicubic',
        'convert':      'RGB',
        'is_npy':       False,
    },
    'depth': {
        'hf_id':        'EPFL-VILAB/4M_tokenizers_depth_8k_224-448',
        'wds_key':      'npy',          # depth is stored as .npy
        'tok_folder':   'tok_depth@224',
        'resample_mode':'bilinear',
        'convert':      None,
        'is_npy':       True,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ShardDataset(IterableDataset):
    def __init__(self, samples, task, task_cfg, n_crops, input_size,
                 center_aug, random_aug, existing_crop_settings):
        self.samples = samples
        self.task = task
        self.task_cfg = task_cfg
        self.n_crops = n_crops
        self.input_size = input_size
        self.center_aug = center_aug
        self.random_aug = random_aug
        self.existing_crop_settings = existing_crop_settings

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process, use all samples
            samples = self.samples
        else:
            # split samples across workers
            per_worker = math.ceil(len(self.samples) / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.samples))
            samples = self.samples[start:end]

        for sample in samples:
            key = sample['__key__']
            try:
                img = load_image(sample, self.task_cfg)
            except Exception as e:
                print(f"Warning: skipping {key}: {e}")
                continue
            existing = self.existing_crop_settings.get(key)
            settings = get_crop_settings(
                key, img, self.task, self.n_crops, self.input_size,
                self.center_aug, self.random_aug, existing_settings=existing
            )
            imgs = apply_crops(
                img, self.task, settings, self.input_size,
                self.task_cfg['resample_mode'], MODALITY_TRANSFORMS_DIVAE
            )
            yield key, settings, imgs  # (n_crops, C, H, W)
    
    def __len__(self):
        return len(self.samples)

def load_image(sample, task_cfg):
    """Load a PIL image from a WebDataset sample."""
    raw = sample.get(task_cfg['wds_key'])
    if raw is None:
        available = [k for k in sample if not k.startswith('__')]
        raise KeyError(
            f"Key '{task_cfg['wds_key']}' not in sample '{sample['__key__']}'. "
            f"Available: {available}"
        )

    if task_cfg['is_npy']:
        arr = np.load(io.BytesIO(raw))
        # Depth is typically (H, W) or (H, W, 1) float — convert to PIL
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        # Normalise to uint16 range for PIL
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 65535).astype(np.uint16)
        img = Image.fromarray(arr, mode='I;16')
    else:
        img = Image.open(io.BytesIO(raw))

    if task_cfg['convert']:
        img = img.convert(task_cfg['convert'])

    return img


def get_crop_settings(key, img, task, n_crops, input_size,
                      center_aug, random_aug, existing_settings=None):
    """
    Return crop settings array of shape (n_crops, 5): (i, j, h, w, h_flip).
    First crop is always a non-flipped center crop.
    Subsequent crops are random but deterministic from the sample key.
    If existing_settings are provided (from a prior RGB pass), reuse them.
    """
    if existing_settings is not None:
        return existing_settings

    # Seed RNG deterministically from key so RGB and depth passes match
    seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
    rng_state = torch.get_rng_state()
    np_state = np.random.get_state()
    torch.manual_seed(seed)
    np.random.seed(seed)

    settings = []
    crop_coords, _, _, _, _ = center_aug({task: img}, None)
    settings.append((*crop_coords, 0))

    for _ in range(1, n_crops):
        crop_coords, h_flip, _, _, _ = random_aug({task: img}, None)
        settings.append((*crop_coords, 1 if h_flip else 0))

    torch.set_rng_state(rng_state)
    np.random.set_state(np_state)

    return np.array(settings, dtype=np.int32)


def apply_crops(img, task, settings, input_size, resample_mode, task_transforms):
    """Apply crop settings to image, return stacked tensor (n_crops, C, H, W)."""
    imgs = []
    for i, j, h, w, h_flip in settings:
        img_mod = task_transforms[task].preprocess(img.copy())
        img_mod = task_transforms[task].image_augment(
            img_mod, (i, j, h, w), h_flip, None,
            (input_size, input_size), None, resample_mode
        )
        img_mod = task_transforms[task].postprocess(img_mod)
        imgs.append(img_mod)
    return torch.stack(imgs)  # (n_crops, C, H, W)


def tokenize_batch(model, imgs_batch, device):
    """
    imgs_batch: (B, C, H, W) tensor
    returns: (B, n_tokens) int16 numpy array
    """
    imgs_batch = imgs_batch.to(device)
    with torch.no_grad():
        tokens = model.tokenize(imgs_batch)          # (B, H', W')
        if tokens.size(-1) == 1:
            tokens = tokens.squeeze(2)
        tokens = rearrange(tokens, 'b h w -> b (h w)')  # (B, n_tokens)
    return tokens.detach().cpu().numpy().astype(np.int16)


def npy_to_bytes(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def shard_name(i):
    return f'shard-{i:05d}.tar'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device)
    task_cfg = TASK_CONFIG[args.task]

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {task_cfg['hf_id']}")
    model = DiVAE.from_pretrained(task_cfg['hf_id']).to(device).eval()
    print("Tokenizer loaded.")

    # --- Augmenters ---
    center_aug = CenterCropImageAugmenter(
        target_size=args.input_size, hflip=0.0, main_domain=args.task
    )
    random_aug = RandomCropImageAugmenter(
        target_size=args.input_size, hflip=0.5,
        crop_scale=(args.min_crop_scale, 1.0),
        crop_ratio=(0.75, 1.3333),
        main_domain=args.task
    )

    # --- Find input shards ---
    input_dir = os.path.join(args.data_root, args.task)
    tar_files = sorted(glob.glob(os.path.join(input_dir, '*.tar')))
    if not tar_files:
        raise FileNotFoundError(f"No tar files found in {input_dir}")
    print(f"Found {len(tar_files)} shards in {input_dir}")

    # --- Output dirs ---
    tok_out_dir = os.path.join(args.data_root, task_cfg['tok_folder'])
    crop_out_dir = os.path.join(args.data_root, 'crop_settings')
    os.makedirs(tok_out_dir, exist_ok=True)
    os.makedirs(crop_out_dir, exist_ok=True)

    # --- Process shard by shard ---
    start_time = time.time()
    total_written = 0

    for shard_idx, tar_path in enumerate(tqdm(tar_files, desc='Shards')):
        shard_file = shard_name(shard_idx)
        tok_out_path = os.path.join(tok_out_dir, shard_file)
        crop_out_path = os.path.join(crop_out_dir, shard_file)

        if os.path.exists(tok_out_path) and not args.force_retokenize:
            print(f"Skipping {shard_file} (already exists)")
            continue

        # Load existing crop settings shard if available (written by RGB pass)
        existing_crop_settings = {}
        if os.path.exists(crop_out_path):
            try:
                for s in wds.WebDataset(crop_out_path):
                    key = s['__key__']
                    existing_crop_settings[key] = np.load(io.BytesIO(s['npy']))
            except Exception as e:
                print(f"Warning: could not load crop settings from {crop_out_path}: {e}")

        # Collect all samples from this shard
        dataset = wds.WebDataset(tar_path)
        samples = list(dataset)  # load full shard into memory (1k-10k images)
        print(len(samples))

        if args.dryrun:
            for s in samples[:3]:
                print(f"  dryrun: key={s['__key__']} -> {tok_out_path}")
            continue

        # Write output tars
        tok_writer = wds.TarWriter(tok_out_path)
        crop_writer = wds.TarWriter(crop_out_path) if not os.path.exists(crop_out_path) else None

        shard_ds = ShardDataset(
            samples, args.task, task_cfg, args.n_crops,
            args.input_size, center_aug, random_aug, existing_crop_settings
        )
        loader = DataLoader(
            shard_ds,
            batch_size=args.batch_size_dataloader,
            num_workers=4,
            prefetch_factor=2,
        )

        n_batches = math.ceil(len(shard_ds) / args.batch_size_dataloader)

        for keys, settings_batch, imgs_batch in tqdm(loader, total=n_batches, desc=f'  {shard_file}', leave=False):
            # imgs_batch: (B, n_crops, C, H, W)
            imgs_flat = rearrange(imgs_batch, 'b n c h w -> (b n) c h w')
            sub_batches = imgs_flat.split(args.batch_size, dim=0)
            all_tokens = np.concatenate([
                tokenize_batch(model, sb, device) for sb in sub_batches
            ])  # (B*n_crops, n_tokens)
            all_tokens = rearrange(all_tokens, '(b n) d -> b n d', n=args.n_crops)

            for key, settings, tokens in zip(keys, settings_batch, all_tokens):
                tok_writer.write({
                    '__key__': key,
                    'npy': npy_to_bytes(tokens),
                })
                if crop_writer is not None:
                    crop_writer.write({
                        '__key__': key,
                        'npy': npy_to_bytes(settings.numpy()),
                    })
        # flush_batch()

        tok_writer.close()
        if crop_writer is not None:
            crop_writer.close()

        total_written += len(samples)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Done. {total_written} samples written in {elapsed}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root dir containing modality subfolders')
    parser.add_argument('--task', type=str, required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument('--n_crops', type=int, default=4,
                        help='Number of crops per image (1=center only, >1 adds random crops)')
    parser.add_argument('--min_crop_scale', type=float, default=0.2)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='GPU batch size for tokenizer inference')
    parser.add_argument('--batch_size_dataloader', type=int, default=512,
                        help='Number of images to accumulate before flushing to GPU')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--force_retokenize', action='store_true', default=False)
    parser.add_argument('--dryrun', action='store_true', default=False)
    args = parser.parse_args()
    main(args)