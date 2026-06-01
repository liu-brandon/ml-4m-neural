"""
Chinchilla Scaling Law Pipeline for 4M Neural Vision Model
=============================================================
Two-part pipeline:
  1. Launch training sweep across (dim, total_tokens) grid
  2. Fit scaling law to collected results using chinchilla package

Install:
    pip install git+https://github.com/kyo-takano/chinchilla.git
    pip install pyyaml

Usage:
    # Run sweep (launches jobs sequentially; adapt for Modal/SLURM parallelism)
    python scaling_sweep.py --mode sweep

    # Fit scaling law after sweep completes
    python scaling_sweep.py --mode fit

    # Run both
    python scaling_sweep.py --mode all
"""

import os
import re
import shutil
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy

import numpy as np


# ─────────────────────────────────────────────
# 1. SWEEP GRID
# ─────────────────────────────────────────────

# Model size axis: (dim, num_heads) — keep head_dim=64 fixed
# MODEL_CONFIGS = [
#     {"dim": 128, "num_heads": 2, "layers": 2},   # ~7.5M params
#     {"dim": 192, "num_heads": 3, "layers": 2},   # ~11M  params
#     {"dim": 256, "num_heads": 4, "layers": 2},   # ~16M  params
#     # {"dim": 256, "num_heads": 4, "layers": 3},
#     # {"dim": 512, "num_heads": 4, "layers": 3}, # 48 million params
# ]
MODEL_CONFIGS = [
    # {"dim": 512, "num_heads": 8, "layers": 3},
    # {"dim": 128, "num_heads": 2, "layers": 2}, 
    {"dim": 256, "num_heads": 4, "layers": 3},
    # {"dim": 192, "num_heads": 3, "layers": 2},
    # {"dim": 256, "num_heads": 4, "layers": 2},
]

# Data axis: total tokens seen in billions
TRAIN_TOKEN_CONFIGS = [5.0]
# TOKEN_CONFIGS = [0.5, 2.0, 5.0]  # B tokens
TOKEN_CONFIGS = [5.0]
# TOKEN_CONFIGS = [8]

# Condition: "rgb_only" | "pixel_meg" | "pixel_eeg"
# Set via --condition CLI flag (or modal_train.py condition param).
# Controls which model/data configs are used and which output subdirectory is written.
CONDITION = "rgb_only"

# ─────────────────────────────────────────────
# 2. BASE CONFIG PATHS  (keyed by condition)
# ─────────────────────────────────────────────
_CFG_BASE = "/opt/repo/ml-4m/cfgs/neural/4m/modal/model"
CONDITION_MODEL_CFGS = {
    # Primary conditions
    "rgb_only": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling.yaml",
    },
    "pixel_meg": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-meg.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-meg.yaml",
    },
    "pixel_eeg": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-eeg.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-eeg.yaml",
    },
    # Subconditions
    # rgb_only_pure_all2all: no directed rgb→depth bias; pure all2all for both CC12M and
    # THINGS — fairer baseline vs neural arms which cannot exploit the rgb→depth shortcut.
    "rgb_only_pure_all2all": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-pure-all2all.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-pure-all2all.yaml",
    },
    # pixel_meg_rvq0: MEG arm with only rvq0; removes rvq1-3 whose near-random gradient
    # signal may be hurting scaling for larger models.
    "pixel_meg_rvq0": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-meg-rvq0.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-meg-rvq0.yaml",
    },
    # pixel_meg_shuffled: null ablation — same config as pixel_meg but neural tokens are
    # shuffled across the batch each forward pass, breaking image-neural correspondence.
    # If training loss still decreases, the signal comes from token statistics, not pairing.
    "pixel_meg_shuffled": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-meg.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-meg.yaml",
        "shuffle_neural": True,
    },
    "pixel_meg_rvq0_shuffled": {
        "2layer": f"{_CFG_BASE}/4m-neural-2e-2d-scaling-meg-rvq0.yaml",
        "3layer": f"{_CFG_BASE}/4m-neural-3e-3d-scaling-meg-rvq0.yaml",
        "shuffle_neural": True,
    },
}

BASE_MODEL_CFG        = CONDITION_MODEL_CFGS[CONDITION]["2layer"]
BASE_MODEL_CFG_3_LAYERS = CONDITION_MODEL_CFGS[CONDITION]["3layer"]

SWEEP_BASE_DIR   = Path("/project/data/scaling_sweep")
SWEEP_OUTPUT_DIR = SWEEP_BASE_DIR / CONDITION
RESULTS_FILE     = SWEEP_OUTPUT_DIR / "results.json"

# Run directory name pattern — used by auto-discovery (independent of MODEL_CONFIGS)
_RUN_RE = re.compile(r"^dim(\d+)_layer(\d+)_tok([\d.]+)B$")


# ─────────────────────────────────────────────
# 3. CONFIG GENERATION
# ─────────────────────────────────────────────

def make_run_name(dim, layers, total_tokens):
    return f"dim{dim}_layer{layers}_tok{total_tokens}B" # change this back
    # return f"dim{dim}_tok{total_tokens}B"


def generate_configs(dim, layers, num_heads, total_tokens, sweep_output_dir, test_run,
                     large_gpu=False):
    """Generate patched yaml configs for a single sweep run."""
    run_name = make_run_name(dim, layers, total_tokens)
    run_dir  = sweep_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if layers == 3:
        model_cfg_file = BASE_MODEL_CFG_3_LAYERS
    else:
        model_cfg_file = BASE_MODEL_CFG

    # --- Model config ---
    with open(model_cfg_file) as f:
        model_cfg = yaml.safe_load(f)

    # model_cfg["dim"]       = dim
    # model_cfg["num_heads"] = num_heads
    model_cfg["model"] += f"_{dim}_dim"

    # Per-dim batch sizes tuned for A100-40GB.
    # Memory ≈ bs × (C_fixed + C_act × dim × layers); C_fixed dominates for small models.
    # dim=128 2e2d: 640 is fine (observed 38 GB reserved, ~18 GB actual working set).
    # dim=192 2e2d: 640 likely OOMs (no prior override existed here).
    # dim=256 3e3d: 640 is borderline; 512 gives ~8 GB headroom.
    # dim=512 3e3d: OOMs at bs=512 on 40GB; use --large_gpu for 80GB A100 (bs=512 safe).
    bs_map = {
        (128, 2): 640,
        (192, 2): 512,
        (256, 3): 512,
        (512, 3): 256,
    }
    if large_gpu:
        bs_map[(512, 3)] = 512
    model_cfg["batch_size"] = bs_map.get((dim, layers), 512)

    # clip_grad scales with model size: 512-dim gradients have intrinsically higher L2 norm
    # (~√N scaling) and at bs=256 were clipped 44-75% of training steps. 6.0 is a safety net
    # only; 192/256 dims will never approach it in normal training.
    if large_gpu and (dim, layers) == (512, 3):
        model_cfg["clip_grad"] = 6.0

    if test_run:
        run_dir = run_dir / "test_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_cfg["epoch_size"] = 1024
        model_cfg["warmup_tokens"] = 0.000001
        path_config = Path(model_cfg["data_config"])
        model_cfg["data_config"] = str(path_config.parent / (path_config.stem + "-tiny.yaml"))

    model_cfg_path = run_dir / "model.yaml"

    # --- Training config ---
    # with open(BASE_TRAIN_CFG) as f:
    #     train_cfg = yaml.safe_load(f)
    if test_run:
        total_tokens = 0.00001
    model_cfg["total_tokens"] = total_tokens  # in billions
    model_cfg["output_dir"]   = str(run_dir / "checkpoints")
    with open(model_cfg_path, "w") as f:
        yaml.dump(model_cfg, f)

    # train_cfg_path = run_dir / "train.yaml"
    # with open(train_cfg_path, "w") as f:
    #     yaml.dump(train_cfg, f)

    return model_cfg_path, run_dir


# ─────────────────────────────────────────────
# 4. TRAINING LAUNCH
# ─────────────────────────────────────────────

def launch_training(model_cfg_path, run_dir, shuffle_neural=False):
    """
    Launch a single training run. Adapt this for your cluster/Modal setup.
    Currently runs sequentially via subprocess — see Modal section below.
    """
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        sys.executable,
        "/opt/repo/4m_training/lib/train_4m.py",
        "train",
        "--config", str(model_cfg_path),
    ]
    extra = []
    if shuffle_neural:
        extra.append("--shuffle_neural_tokens")
    if test_run or extra:
        cmd.append("--")
        if test_run:
            extra = ["--device", "cpu", "--dist_backend", "gloo"] + extra
        cmd.extend(extra)

    print(f"Launching: {' '.join(cmd)}")
    print(f"Logging to: {log_path}", flush=True)

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        proc.wait()
    result_returncode = proc.returncode

    if result_returncode != 0:
        print(f"WARNING: run exited with code {result_returncode}, check {log_path}")

    return result_returncode

# ─────────────────────────────────────────────
# 5. RESULT COLLECTION
# ─────────────────────────────────────────────

# def count_parameters(dim, num_heads):
#     """
#     Estimate total parameter count for your architecture.
#     Adjust vocab sizes to match your actual tokenizers.
#     """
#     vocab_rgb   = 16384
#     vocab_depth = 8192
#     # vocab_neural = 8192  # uncomment for 3-modality condition

#     encoder_depth = 2
#     decoder_depth = 2
#     mlp_ratio     = 4

#     # Embedding parameters (encoder + decoder, with decoder tied in/out)
#     emb_params = (
#         (vocab_rgb   + vocab_depth) * dim * 2   # encoder emb + decoder emb (tied in/out)
#         # + vocab_neural * dim * 2               # add for 3-modality
#     )

#     # Transformer block parameters
#     blocks = encoder_depth + decoder_depth
#     attn_params = 4 * dim * dim                  # Q, K, V, O per block
#     ffn_params  = 2 * mlp_ratio * dim * dim      # FFN up + down per block
#     transformer_params = blocks * (attn_params + ffn_params)

#     # Misc: norms, mod_emb, proj, mask_token
#     misc_params = dim * 20

#     return emb_params + transformer_params + misc_params


# cc12m keys are required for a row to be included; things keys collected when present
_CC12M_REQUIRED = [
    "[Fixed Eval (cc12m)] tok_rgb@224_loss",
    "[Fixed Eval (cc12m)] tok_depth@224_loss",
]
_THINGS_OPTIONAL_PATTERNS = [
    "[Fixed Eval (things)] tok_rgb@224_loss",
    "[Fixed Eval (things)] tok_depth@224_loss",
    "[Fixed Eval (things)] tok_meg_rvq0_loss",
    "[Fixed Eval (things)] tok_meg_rvq1_loss",
    "[Fixed Eval (things)] tok_meg_rvq2_loss",
    "[Fixed Eval (things)] tok_meg_rvq3_loss",
    "[Fixed Eval (things)] tok_eeg_loss",
]


def extract_final_loss(run_dir, test_run=False):
    """Parse checkpoints/log.txt and return rows that have Fixed Eval (cc12m) data.

    Rows without both cc12m RGB and depth eval keys are skipped (they are
    training-only log lines that never triggered an eval).  Any things Fixed Eval
    keys found in the same row are included as optional fields.
    """
    log_path = run_dir / "checkpoints" / "log.txt"
    if not log_path.exists():
        print(f"  No log found at {log_path}")
        return []

    losses = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "total_tokens_seen_b" not in data:
                continue
            if not all(k in data for k in _CC12M_REQUIRED):
                continue
            result = {"total_tokens_seen_b": data["total_tokens_seen_b"]}
            for k in _CC12M_REQUIRED:
                result[k] = data[k]
            for k in _THINGS_OPTIONAL_PATTERNS:
                if k in data:
                    result[k] = data[k]
            losses.append(result)

    if not losses:
        print(f"  Could not parse Fixed Eval loss from {log_path}")
    return losses

def extract_model_size(run_dir, test_run):
    """Total parameter count from train.log (includes embedding tables)."""
    log_path = run_dir / "train.log"
    if not log_path.exists():
        print(f"  No log found at {log_path}")
        return None
    with open(log_path) as f:
        for line in f:
            if "Number of params:" in line:
                return float(line.split(": ")[1].split(" ")[0]) * 1e6


def transformer_param_count(dim: int, layers: int) -> int:
    """Transformer backbone parameters, excluding embedding tables.

    Embedding tables (vocab × dim per modality) are O(dim) lookup ops, not O(dim²)
    compute like attention/FFN. Including them inflates N for small models and makes
    cross-condition comparisons misleading — each condition has different embedding sizes
    (rgb_only < pixel_eeg < pixel_meg) for the same backbone.

    Each encoder block (no bias): 4*dim² (attn) + 8*dim² (FFN, gelu 4×) = 12*dim²
    Each decoder block (no bias): 4*dim² (cross-attn) + 4*dim² (self-attn) + 8*dim² = 16*dim²
    Layer norms O(dim) are negligible vs O(dim²).
    4M uses equal encoder/decoder depth (ne == nd == layers).
    """
    return (12 + 16) * layers * dim ** 2

def _pick_token_checkpoints(losses: list[dict], n_points: int = 3, min_tokens: float = 1.0) -> list[dict]:
    """Return up to n_points rows at evenly-spaced token fractions of the run's budget.

    Picks targets at 1/n, 2/n, …, n/n of the final token count, then snaps each
    to the nearest available checkpoint row.  Deduplicates so the same row is never
    returned twice (can happen when checkpointing was sparse).  Rows below min_tokens
    are excluded so the Chinchilla fit only uses well-trained checkpoints.
    """
    if not losses:
        return []
    max_tokens = max(r["total_tokens_seen_b"] for r in losses)
    targets = [(i + 1) / n_points * max_tokens for i in range(n_points)]
    seen: set[float] = set()
    picked = []
    for target in targets:
        if target < min_tokens:
            continue
        best = min(losses, key=lambda r: abs(r["total_tokens_seen_b"] - target))
        t = best["total_tokens_seen_b"]
        if t not in seen:
            seen.add(t)
            picked.append(best)
    return picked


def _discover_runs_in_dir(condition_dir: Path, test_run: bool = False):
    """Yield (dim, layers, total_tokens, run_dir) for each matching subdir."""
    for d in sorted(condition_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        dim, layers, tok = int(m.group(1)), int(m.group(2)), float(m.group(3))
        run_dir = d / "test_run" if test_run else d
        yield dim, layers, tok, run_dir


def collect_results(sweep_base_dir: Path, conditions=None, test_run: bool = False):
    """Collect (N, D, loss) for all conditions found under sweep_base_dir.

    Auto-discovers run directories from the filesystem — does not require
    MODEL_CONFIGS or TOKEN_CONFIGS to be set.  conditions can be a list of
    condition names to process, or None to auto-discover every subdirectory.
    Saves {sweep_base_dir}/{condition}/results.json per condition.
    """
    sweep_base_dir = Path(sweep_base_dir)

    if conditions is None:
        conditions = [d.name for d in sorted(sweep_base_dir.iterdir())
                      if d.is_dir() and not d.name.startswith(".")]
    elif isinstance(conditions, str):
        conditions = [conditions]

    all_results: dict[str, list] = {}

    for condition in conditions:
        condition_dir = sweep_base_dir / condition
        if not condition_dir.exists():
            print(f"  Skipping {condition}: directory not found")
            continue

        print(f"\nCollecting condition: {condition}")
        results = []

        for dim, layers, total_tokens, run_dir in _discover_runs_in_dir(condition_dir, test_run):
            run_name = make_run_name(dim, layers, total_tokens)
            losses   = extract_final_loss(run_dir, test_run)
            N        = transformer_param_count(dim, layers)
            N_total  = extract_model_size(run_dir, test_run)

            checkpoints = _pick_token_checkpoints(losses)
            if checkpoints:
                for loss in checkpoints:
                    entry = {
                        "run":          run_name,
                        "dim":          dim,
                        "N":            N,
                        "N_total":      N_total,
                        "layers":       layers,
                        "total_tokens": loss["total_tokens_seen_b"],
                        "D":            loss["total_tokens_seen_b"] * 1e9,
                        "loss_rgb":     loss["[Fixed Eval (cc12m)] tok_rgb@224_loss"],
                        "loss_depth":   loss["[Fixed Eval (cc12m)] tok_depth@224_loss"],
                    }
                    # Optional neural losses — present for pixel_meg / pixel_eeg conditions
                    _neural_map = {
                        "loss_meg_rvq0": "[Fixed Eval (things)] tok_meg_rvq0_loss",
                        "loss_meg_rvq1": "[Fixed Eval (things)] tok_meg_rvq1_loss",
                        "loss_meg_rvq2": "[Fixed Eval (things)] tok_meg_rvq2_loss",
                        "loss_meg_rvq3": "[Fixed Eval (things)] tok_meg_rvq3_loss",
                        "loss_eeg":      "[Fixed Eval (things)] tok_eeg_loss",
                        "loss_things_rgb":   "[Fixed Eval (things)] tok_rgb@224_loss",
                        "loss_things_depth": "[Fixed Eval (things)] tok_depth@224_loss",
                    }
                    for out_key, src_key in _neural_map.items():
                        if src_key in loss:
                            entry[out_key] = loss[src_key]
                    results.append(entry)
                tokens_str = ", ".join(f"{l['total_tokens_seen_b']:.2f}B" for l in checkpoints)
                print(f"  {run_name}: {len(checkpoints)} points at [{tokens_str}]")
            else:
                print(f"  {run_name}: INCOMPLETE — skipping")

        results_file = condition_dir / "results.json"
        condition_dir.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} points → {results_file}")
        all_results[condition] = results

    return all_results


# ─────────────────────────────────────────────
# 6. SCALING LAW FIT
# ─────────────────────────────────────────────

def fit_scaling_law(results, sweep_output_dir, condition, loss_type="rgb"):
    """
    Fit L(N, D) = E + A/N^alpha + B/D^beta using the chinchilla package.
 
    Key API facts from source:
      - cc.append(N, D, loss) adds a data point to the internal database
      - D is number of IMAGE SAMPLES, not tokens — convert from tokens
      - cc.fit() runs BFGS over param_grid to find best (E, A, B, alpha, beta)
      - cc.report() prints fitted params and goodness of fit
      - cc.predict_loss(N, D) gives predicted loss after fitting
    """
    try:
        from chinchilla import Chinchilla
    except ImportError:
        print("Install chinchilla: pip install git+https://github.com/kyo-takano/chinchilla.git")
        return
 
    if len(results) < 3:
        print(f"Only {len(results)} results — chinchilla needs at least 3 to fit")
        return
 
    print(f"\nFitting scaling law on {len(results)} points...")
 
    N_values    = [r["N"]    for r in results]
    # D in chinchilla is samples, not tokens
    # Convert: tokens / (num_input_tokens + num_target_tokens) = samples seen
    # Use total_tokens * 1e9 / tokens_per_sample as effective sample count
    # If you don't know tokens_per_sample exactly, use raw token count — 
    # it shifts B but not beta, which is what you care about comparatively
    if loss_type == "rgb":
        loss_key = "loss_rgb"
    elif loss_type == "depth":
        loss_key = "loss_depth"
    else:
        raise RuntimeError(f"Bad loss type: {loss_type}")
    D_values    = [r["D"]    for r in results]  # already in raw tokens from collect_results
    loss_values = [r[loss_key] for r in results]
 
    print(f"  N range:    {min(N_values):,} – {max(N_values):,}")
    print(f"  D range:    {min(D_values):,.0f} – {max(D_values):,.0f}")
    print(f"  Loss range: {min(loss_values):.4f} – {max(loss_values):.4f}")
 
    # Initialize Chinchilla — no seed_ranges needed since we're only fitting,
    # not using it to suggest next runs
    db_dir = sweep_output_dir / f"chinchilla_db_{loss_type}"
    if db_dir.exists():
        shutil.rmtree(db_dir)

    cc = Chinchilla(
        project_dir=str(db_dir),
 
        param_grid=dict(
            # E: irreducible loss floor — set below your best observed loss
            # Your epoch 27 loss was ~8.0, still improving, so floor ~5-7
            E=np.linspace(6.0, 9, 5),
 
            # A, B: penalty coefficients — scale with your loss magnitude (~8-9)
            # With N~11M and alpha~0.2: A/N^0.2 ~ A/27, so A~hundreds to be meaningful
            A=np.linspace(100, 2200, 5),
            B=np.linspace(100, 2200, 5),
 
            # alpha, beta: scaling exponents
            # Vision tends lower than language (Chinchilla found ~0.34 for text)
            # Embedding-heavy models like yours may have even lower alpha
            alpha=np.linspace(0.05, 0.4, 5),
            beta=np.linspace(0.1,  0.5, 5),
        ),
    )
 
    # Load all results into chinchilla's database
    # append() signature from source: cc.append(N=N, D=D, loss=loss)
    for r in results:
        cc.append(N=r["N"], D=r["D"], loss=r[loss_key])
 
    print(f"\nLoaded {len(results)} data points into chinchilla database")
 
    # Fit — runs BFGS over all param_grid combinations in parallel
    # Minimum 3 points required (enforced in source)
    cc.fit(parallel=True)
 
    # Report — prints fitted params + goodness of fit via loss_fn (asymmetric_mae)
    cc.report(plot=True)  # set plot=False if no display available (e.g. on cluster)
 
    # Save fitted params manually as well
    try:
        params = cc.get_params()
        params["condition"] = condition
        params["n_points"]  = len(results)
        fit_path = sweep_output_dir / "scaling_law_fit.json"
        with open(fit_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"\nFit saved to {fit_path}")
        print(f"\nFitted scaling law:")
        print(f"  L(N,D) = {params['E']:.4f} + {params['A']:.4f}/N^{params['alpha']:.4f} + {params['B']:.4f}/D^{params['beta']:.4f}")
    except ValueError as e:
        print(f"Could not save params: {e}")
 
    return cc


# ─────────────────────────────────────────────
# 7. PLOTTING
# ─────────────────────────────────────────────

def _safe_legend(ax) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


def plot_results(results, sweep_output_dir, model_configs, condition, law=None, loss_type="rgb"):
    """Basic loss curve plots for qualitative scaling observations."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy for plotting")
        return

    loss_key = f"loss_{loss_type}"
    valid = [r for r in results if r.get(loss_key) is not None]
    if not valid:
        print(f"  No results with loss_{loss_type} — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Scaling Laws — {condition}  [{loss_type}]")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    # Left: loss vs D (tokens), one line per (dim, layers)
    ax = axes[0]
    dims_seen = sorted({(r["dim"], r["layers"]) for r in valid})
    for i, (dim, layers) in enumerate(dims_seen):
        pts = sorted(
            (r["D"], r[loss_key])
            for r in valid if r["dim"] == dim and r["layers"] == layers
        )
        if not pts:
            continue
        N = next(r["N"] for r in valid if r["dim"] == dim and r["layers"] == layers)
        D_pts, L_pts = zip(*pts)
        ax.plot(D_pts, L_pts, "o-", color=colors[i % len(colors)],
                label=f"dim={dim} L{layers} (N≈{N/1e6:.1f}M)")

    ax.set_xlabel("Tokens seen (D)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Loss vs. Data Scale")
    ax.set_xscale("log")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    # Right: loss vs N (params), one line per approximate token budget.
    # Snap each result's total_tokens to the nearest 0.5B to group iso-FLOPs curves
    # without requiring exact matches against TOKEN_CONFIGS.
    ax = axes[1]
    token_colors = ["#9C27B0", "#00BCD4", "#FF9800", "#4CAF50"]

    def _snap(t, resolution=0.5):
        return round(t / resolution) * resolution

    snapped_budgets = sorted({_snap(r["total_tokens"]) for r in valid})
    for i, budget in enumerate(snapped_budgets):
        pts = sorted(
            (r["N"], r[loss_key])
            for r in valid if _snap(r["total_tokens"]) == budget
        )
        if not pts:
            continue
        pts_deduped = dict(pts)  # keep lowest loss if multiple runs share same N
        N_pts, L_pts = zip(*sorted(pts_deduped.items()))
        ax.plot(N_pts, L_pts, "s-", color=token_colors[i % len(token_colors)],
                label=f"~{budget:.1f}B tokens")

    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Loss vs. Model Scale")
    ax.set_xscale("log")
    _safe_legend(ax)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = sweep_output_dir / "scaling_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def run_sweep(model_configs, train_token_configs, sweep_output_dir, condition, test_run=False,
              large_gpu=False):
    """Launch all training runs sequentially."""
    print(f"Starting sweep: {len(model_configs)} model sizes x {len(train_token_configs)} token counts")
    print(f"Condition: {condition}  large_gpu: {large_gpu}\n")

    condition_meta = CONDITION_MODEL_CFGS.get(condition, {})
    shuffle_neural = condition_meta.get("shuffle_neural", False)
    if shuffle_neural:
        print("[shuffle] Neural token shuffling enabled for this condition\n")

    for model_cfg in model_configs:
        for total_tokens in train_token_configs:
            dim       = model_cfg["dim"]
            num_heads = model_cfg["num_heads"]
            layers = model_cfg["layers"]
            run_name  = make_run_name(dim, layers, total_tokens)

            print(f"\n{'='*50}")
            print(f"Run: {run_name}  (dim={dim}, {total_tokens}B tokens)")
            print(f"{'='*50}")

            model_cfg_path, run_dir = generate_configs(
                dim, layers, num_heads, total_tokens, sweep_output_dir, test_run,
                large_gpu=large_gpu,
            )
            launch_training(model_cfg_path, run_dir, shuffle_neural=shuffle_neural)

    print("\nSweep complete. Run with --mode fit to analyze results.")


def run_fit(sweep_base_dir: Path, conditions=None, test_run: bool = False, loss_type: str = "rgb"):
    """Load results.json for each condition and fit + plot the scaling law."""
    sweep_base_dir = Path(sweep_base_dir)

    if conditions is None:
        conditions = [d.name for d in sorted(sweep_base_dir.iterdir())
                      if d.is_dir() and not d.name.startswith(".")]
    elif isinstance(conditions, str):
        conditions = [conditions]

    for condition in conditions:
        results_file = sweep_base_dir / condition / "results.json"
        if not results_file.exists():
            print(f"  {condition}: no results.json — run collect first")
            continue

        print(f"\nFitting condition: {condition}")
        with open(results_file) as f:
            results = json.load(f)

        if not results:
            print(f"  {condition}: empty results")
            continue

        valid = [r for r in results if r.get(f"loss_{loss_type}") is not None]
        if len(valid) < 3:
            print(f"  {condition}: only {len(valid)} valid {loss_type} points — need ≥3")
            continue

        sweep_output_dir = sweep_base_dir / condition
        law = fit_scaling_law(valid, sweep_output_dir, condition, loss_type)
        plot_results(valid, sweep_output_dir, MODEL_CONFIGS, condition, law, loss_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sweep", "fit", "collect", "all"],
        default="all",
        help="sweep: run training; fit: analyze results; collect: just parse logs; all: both"
    )
    parser.add_argument(
        "--condition",
        choices=list(CONDITION_MODEL_CFGS.keys()),
        default=None,
        help="which modality arm; omit to process all conditions found under sweep_base_dir "
             "(collect/fit) or required for sweep/dryrun (controls model/data configs)",
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--loss_type",
        choices=["rgb", "depth", "meg_rvq0", "meg_rvq1", "meg_rvq2", "meg_rvq3",
                 "eeg", "things_rgb", "things_depth"],
        type=str,
        default="rgb",
    )
    parser.add_argument(
        "--large_gpu",
        action="store_true",
        default=False,
        help="use A100-80GB batch sizes (dim=512 3e3d: bs=512 instead of 256)",
    )
    args = parser.parse_args()
    test_run = args.test_run
    loss_type = args.loss_type
    large_gpu = args.large_gpu

    # sweep / all require a specific condition (controls which model+data configs to launch).
    # collect / fit default to all conditions when --condition is omitted.
    CONDITION = args.condition

    if args.mode in ("sweep", "all"):
        if CONDITION is None:
            parser.error("--condition is required for --mode sweep / all")
        BASE_MODEL_CFG = CONDITION_MODEL_CFGS[CONDITION]["2layer"]
        BASE_MODEL_CFG_3_LAYERS = CONDITION_MODEL_CFGS[CONDITION]["3layer"]
        SWEEP_OUTPUT_DIR = SWEEP_BASE_DIR / CONDITION

    if args.mode == "sweep":
        run_sweep(MODEL_CONFIGS, TRAIN_TOKEN_CONFIGS, SWEEP_OUTPUT_DIR, CONDITION, test_run,
                  large_gpu=large_gpu)
    elif args.mode == "fit":
        run_fit(SWEEP_BASE_DIR, conditions=CONDITION, test_run=test_run, loss_type=loss_type)
    elif args.mode == "collect":
        collect_results(SWEEP_BASE_DIR, conditions=CONDITION, test_run=test_run)
    elif args.mode == "all":
        run_sweep(MODEL_CONFIGS, TRAIN_TOKEN_CONFIGS, SWEEP_OUTPUT_DIR, CONDITION, test_run)
        collect_results(SWEEP_BASE_DIR, conditions=CONDITION, test_run=test_run)
        run_fit(SWEEP_BASE_DIR, conditions=CONDITION, test_run=test_run, loss_type=loss_type)