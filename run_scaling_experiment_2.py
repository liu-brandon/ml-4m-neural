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
    {"dim": 512, "num_heads": 8, "layers": 3},
    {"dim": 256, "num_heads": 4, "layers": 3},
    {"dim": 192, "num_heads": 3, "layers": 2},
    {"dim": 128, "num_heads": 2, "layers": 2},
    # {"dim": 256, "num_heads": 4, "layers": 2},
]

# Data axis: total tokens seen in billions
TRAIN_TOKEN_CONFIGS = [5.0]
# TOKEN_CONFIGS = [0.5, 2.0, 5.0]  # B tokens
TOKEN_CONFIGS = [5.0]
# TOKEN_CONFIGS = [8]

# Condition: set to "rgb_only" or "rgb_neural"
# Run sweep twice, once per condition, to compare scaling laws
CONDITION = "rgb_only"  # change to "rgb_neural" for experimental condition

# ─────────────────────────────────────────────
# 2. BASE CONFIG PATHS
# ─────────────────────────────────────────────
# These are your existing yaml configs — sweep will copy and patch them

BASE_MODEL_CFG = "/opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-2e-2d-scaling.yaml"
BASE_MODEL_CFG_3_LAYERS = "/opt/repo/ml-4m/cfgs/neural/4m/modal/model/4m-neural-3e-3d-scaling.yaml"
# BASE_TRAIN_CFG = "cfgs/neural/4m/training/base_train.yaml"
BASE_DATA_CFG  = "/opt/repo/ml-4m/cfgs/neural/4m/modal/data/rgb-depth-a0.5.yaml"  # swap for rgb-depth-neural.yaml

# SWEEP_OUTPUT_DIR = Path(f"/scratch/users/liubr/neural-image-foundation-data/scaling_sweep/{CONDITION}")
SWEEP_OUTPUT_DIR = Path("/project/data/scaling_sweep")
RESULTS_FILE     = SWEEP_OUTPUT_DIR / "results.json"


# ─────────────────────────────────────────────
# 3. CONFIG GENERATION
# ─────────────────────────────────────────────

def make_run_name(dim, layers, total_tokens):
    return f"dim{dim}_layer{layers}_tok{total_tokens}B" # change this back
    # return f"dim{dim}_tok{total_tokens}B"


def generate_configs(dim, layers, num_heads, total_tokens, sweep_output_dir, test_run):
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

def launch_training(model_cfg_path, run_dir):
    """
    Launch a single training run. Adapt this for your cluster/Modal setup.
    Currently runs sequentially via subprocess — see Modal section below.
    """
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    cmd = [
        "torchrun", "--nproc_per_node=1", "/opt/repo/ml-4m/run_training_4m.py",             # your main training entrypoint
        "--config",      str(model_cfg_path)
    ]
    if test_run:
        cmd.extend([
            "--device", "cpu",
            "--dist_backend", "gloo"
        ])

    print(f"Launching: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")

    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"WARNING: run exited with code {result.returncode}, check {log_path}")

    return result.returncode

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


def extract_final_loss(run_dir, test_run=False):
    """
    Extract final validation loss from training log or checkpoint.
    Adapt to however your training code logs results.

    Returns float loss or None if run didn't complete.
    """
    log_path = run_dir / "checkpoints" / "log.txt"

    if not log_path.exists():
        print(f"  No log found at {log_path}")
        return []

    # Parse the last eval loss line from your log format:
    # "[Eval (cc12m)]  ... loss: X.XXXX ..."
    losses = [] # each a dict of (rgb loss, depth loss, tokens seen)
    with open(log_path) as f:
        for line in f:
            data = json.loads(line)
            result = {
                "total_tokens_seen_b": data.get("total_tokens_seen_b"),
            }
            
            # Extract all Fixed Eval tok_rgb@224_loss and tok_depth@224_loss
            for key, value in data.items():
                if "Fixed Eval" in key and ("tok_rgb@224_loss" in key or "tok_depth@224_loss" in key):
                    result[key] = value
            losses.append(result)
    

    if len(losses) == 0:
        print(f"  Could not parse loss from {log_path}")

    return losses

def extract_model_size(run_dir, test_run):
    log_path = run_dir / "train.log"
    

    if not log_path.exists():
        print(f"  No log found at {log_path}")
        return None
    
    with open(log_path) as f:
        for line in f:
            if "Number of params:" in line:
                return float(line.split(": ")[1].split(" ")[0]) * 1e6

def collect_results(results_file, sweep_output_dir, model_configs, token_configs, test_run=False):
    """Collect (N, D, loss) from all completed runs."""
    results = []

    for model_cfg in model_configs:
        dim       = model_cfg["dim"]
        num_heads = model_cfg["num_heads"]
        # N         = count_parameters(dim, num_heads)
        layers    = model_cfg["layers"]

        for total_tokens in token_configs:
            run_name = make_run_name(dim, layers, total_tokens)
            run_dir  = sweep_output_dir / run_name
            if test_run:
                run_dir = run_dir / "test_run"

            # D    = int(total_tokens * 1e9)   # convert B tokens → raw count
            losses = extract_final_loss(run_dir, test_run)
            N = extract_model_size(run_dir, test_run)

            if len(losses) > 0:
                indices = set([
                    len(losses) // 3,
                    2 * len(losses) // 3,
                    len(losses) - 1,
                ])
                for i, loss in enumerate(losses):
                    if i not in indices:
                        continue
                    total_tokens_seen = loss["total_tokens_seen_b"]
                    if total_tokens_seen < 1.0:
                        continue

                    results.append({
                        "run":          run_name,
                        "dim":          dim,
                        "N":            N,
                        "layers":       layers,
                        "total_tokens": loss["total_tokens_seen_b"],
                        "D":            loss["total_tokens_seen_b"] * 1e9,
                        "loss_rgb":     loss["[Fixed Eval (cc12m)] tok_rgb@224_loss"],
                        "loss_depth":   loss["[Fixed Eval (cc12m)] tok_depth@224_loss"]
                    })
                # print(f"  {run_name}: N={N:,}  D={D:,}  losses={losses:}")
            else:
                print(f"  {run_name}: INCOMPLETE — skipping")

    # Save results
    sweep_output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {results_file}")
    return results


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
    cc = Chinchilla(
        project_dir=str(sweep_output_dir / f"chinchilla_db_{loss_type}"),
 
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

def plot_results(results, sweep_output_dir, model_configs, condition, law=None, loss_type="rgb"):
    """Basic loss curve plots for qualitative scaling observations."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("pip install matplotlib numpy for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Scaling Laws — {condition}")

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    # Left: loss vs D (tokens), one line per model size
    ax = axes[0]
    for i, model_cfg in enumerate(model_configs):
        dim = model_cfg["dim"]
        layers = model_cfg["layers"]
        pts = [(r["D"], r[f"loss_{loss_type}"]) for r in results if r["dim"] == dim]
        for r in results:
            if r["dim"] == dim and r["layers"] == layers:
                N = r["N"]
                break

        if not pts:
            continue
        pts.sort()
        D_pts, L_pts = zip(*pts)
        # N = count_parameters(dim, model_cfg["num_heads"])
        ax.plot(D_pts, L_pts, "o-", color=colors[i], label=f"dim={dim} (N≈{N/1e6:.1f}M)")

    ax.set_xlabel("Tokens seen (D)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Loss vs. Data Scale")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: loss vs N (params), one line per token budget
    ax = axes[1]
    token_colors = ["#9C27B0", "#00BCD4", "#FF9800"]
    for i, total_tokens in enumerate(TOKEN_CONFIGS):
        pts = [
            (r["N"],
             r[f"loss_{loss_type}"])
            for r in results if r["total_tokens"] == total_tokens
        ]
        if not pts:
            continue
        pts.sort()
        N_pts, L_pts = zip(*pts)
        ax.plot(N_pts, L_pts, "s-", color=token_colors[i], label=f"{total_tokens}B tokens")

    ax.set_xlabel("Parameters (N)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Loss vs. Model Scale")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = sweep_output_dir / "scaling_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def run_sweep(model_configs, train_token_configs, sweep_output_dir, condition, test_run=False):
    """Launch all training runs sequentially."""
    print(f"Starting sweep: {len(model_configs)} model sizes x {len(train_token_configs)} token counts")
    print(f"Condition: {condition}\n")

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
                dim, layers, num_heads, total_tokens, sweep_output_dir, test_run
            )
            launch_training(model_cfg_path, run_dir)

    print("\nSweep complete. Run with --mode fit to analyze results.")


def run_fit(results_file, sweep_output_dir, condition, test_run=False, loss_type="rgb"):
    """Collect results and fit scaling law."""
    # if not results_file.exists():
    #     print("Collecting results from training logs...")
    #     results = collect_results(test_run)
    # else:
    print(f"Loading existing results from {results_file}")
    with open(results_file) as f:
        results = json.load(f)

    if not results:
        print("No results found. Run sweep first, then collect, then fit.")
        return

    law = fit_scaling_law(results, sweep_output_dir, condition, loss_type)
    plot_results(results, sweep_output_dir, MODEL_CONFIGS, condition, law, loss_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sweep", "fit", "collect", "all"],
        default="all",
        help="sweep: run training; fit: analyze results; collect: just parse logs; all: both"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        default=False
    )
    parser.add_argument("--loss_type", choices=["rgb", "depth"], type=str, default="rgb")
    args = parser.parse_args()
    test_run = args.test_run
    loss_type = args.loss_type
    if args.mode == "sweep":
        run_sweep(MODEL_CONFIGS, TRAIN_TOKEN_CONFIGS, SWEEP_OUTPUT_DIR, CONDITION, test_run)
    elif args.mode == "fit":
        run_fit(RESULTS_FILE, SWEEP_OUTPUT_DIR, CONDITION, test_run, loss_type)
    elif args.mode == "collect":
        collect_results(RESULTS_FILE, SWEEP_OUTPUT_DIR, MODEL_CONFIGS, TOKEN_CONFIGS, test_run)
    elif args.mode == "all":
        run_sweep(MODEL_CONFIGS, TRAIN_TOKEN_CONFIGS, SWEEP_OUTPUT_DIR, CONDITION, test_run)
        collect_results(RESULTS_FILE, SWEEP_OUTPUT_DIR, MODEL_CONFIGS, TOKEN_CONFIGS, test_run)
        run_fit(RESULTS_FILE, SWEEP_OUTPUT_DIR, CONDITION, test_run, loss_type)