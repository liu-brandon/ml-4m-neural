"""
modal_scaling_sweep.py
Run the Chinchilla scaling sweep on Modal with persistent storage.

Usage:
    modal run modal_scaling_sweep.py --mode sweep
    modal run modal_scaling_sweep.py --mode fit
    modal run modal_scaling_sweep.py --mode collect
"""

import modal
import argparse

from pathlib import Path

# ─────────────────────────────────────────────
# 1. VOLUME & IMAGE
# ─────────────────────────────────────────────

# Persistent volume — create once, reuse across runs
# First time: modal volume create scaling-sweep-vol
volume = modal.Volume.from_name("project", create_if_missing=True)
VOLUME_MOUNT = "/project"

# Your fourm codebase — mount it into the container
# Assumes you run modal from the root of your fourm repo
# fourm_mount = modal.Mount.from_local_dir(
#     ".",  # local path to your fourm repo root
#     remote_path="/app/ml-4m",
#     condition=lambda p: not any(x in p for x in [
#         "__pycache__", ".git", "*.pyc", 
#         "scaling_sweep_results", "checkpoints", "*.sh", "*.out"
#     ])
# )

image = (
    modal.Image.debian_slim(python_version="3.9.23")
    .add_local_dir(
        ".",  # local fourm repo root
        remote_path="/app/ml-4m",
    )
    .run_commands(
        "pip install -e /app/ml-4m"  # installs your fourm package
    )
)

app = modal.App("scaling-sweep", image=image)


# ─────────────────────────────────────────────
# 2. SWEEP FUNCTION (one run per model config)
# ─────────────────────────────────────────────

@app.function(
    # gpu="L40S",
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 12,  # 12 hours max
    volumes={VOLUME_MOUNT: volume},
    mounts=[fourm_mount],
)
def run_training_job(model_cfg: dict, sweep_output_dir: str, total_tokens: float, condition: str, test_run: bool = False):
    """Run a single training job inside Modal."""
    import sys
    import os
    sys.path.insert(0, "/project/app/ml-4m")
    os.chdir("/project/app/ml-4m")

    from run_scaling_experiment_2 import generate_configs, launch_training

    dim = model_cfg["dim"]
    layers = model_cfg["layers"]
    num_heads = model_cfg["num_heads"]

    print(f"Starting run: dim={dim}, layers={layers}, tokens={total_tokens}B")

    model_cfg_path, run_dir = generate_configs(
        dim, layers, num_heads, total_tokens, sweep_output_dir, test_run
    )
    returncode = launch_training(model_cfg_path, run_dir)

    # Commit writes to volume so they persist
    volume.commit()

    return {
        "dim": dim,
        "layers": layers,
        "total_tokens": total_tokens,
        "returncode": returncode,
        "run_dir": str(run_dir),
    }


# ─────────────────────────────────────────────
# 3. FIT FUNCTION (CPU only, just analysis)
# ─────────────────────────────────────────────

@app.function(
    cpu=4,
    memory=8192,
    timeout=60 * 30,
    volumes={VOLUME_MOUNT: volume},
    mounts=[fourm_mount],
)
def run_fit_job(condition: str, model_configs: str, sweep_output_dir: str, token_configs: str, loss_type: str = "rgb", test_run: bool = False):
    """Run scaling law fit inside Modal."""
    import sys
    import os
    sys.path.insert(0, "/app/ml-4m")
    os.chdir("/app/ml-4m")

    results_file = Path(sweep_output_dir) / "results.json"
    from run_scaling_experiment_2 import collect_results, fit_scaling_law
    import json

    results = collect_results(results_file, sweep_output_dir, model_configs, token_configs, test_run)

    # if not results:
    #     print("No results found.")
    #     return

    law = fit_scaling_law(results_file, sweep_output_dir, condition, loss_type)
    volume.commit()

    return {"n_points": len(results), "condition": condition}


# ─────────────────────────────────────────────
# 4. LOCAL ENTRYPOINT
# ─────────────────────────────────────────────

@app.local_entrypoint()
def main(
    mode: str = "sweep",
    condition: str = "rgb_only",
    loss_type: str = "rgb",
    test_run: bool = False,
):
    # These must match your run_scaling_experiment_2.py
    MODEL_CONFIGS = [
        # {"dim": 512, "num_heads": 8, "layers": 3},
        # {"dim": 256, "num_heads": 4, "layers": 3},
        # {"dim": 192, "num_heads": 3, "layers": 2},
        {"dim": 128, "num_heads": 2, "layers": 2},
        # {"dim": 256, "num_heads": 4, "layers": 2},
    ]
    TRAIN_TOKEN_CONFIGS = [5.0]
    # Import and run your sweep logic
    SWEEP_OUTPUT_DIR = f"/project/data/scaling_sweep/{condition}"
    if mode == "sweep":
        # Launch all jobs in parallel on Modal
        jobs = [
            (model_cfg, total_tokens)
            for model_cfg in MODEL_CONFIGS
            for total_tokens in TRAIN_TOKEN_CONFIGS
        ]
        print(f"Launching {len(jobs)} jobs in parallel...")
        for result in run_training_job.starmap(
            [(cfg, tok, condition, test_run) for cfg, tok in jobs]
        ):
            print(f"Completed: dim={result['dim']}, tokens={result['total_tokens']}B, "
                  f"returncode={result['returncode']}")
    elif mode == "fit":
        result = run_fit_job.remote(condition, MODEL_CONFIGS, SWEEP_OUTPUT_DIR, TRAIN_TOKEN_CONFIGS, loss_type, test_run)
        print(f"Fit complete: {result}")

    # elif mode == "collect":
        # run_fit_job.remote(condition, loss_type, test_run)

    # elif mode == "all":
    #     # Sweep first, then fit
    #     jobs = [
    #         (model_cfg, total_tokens)
    #         for model_cfg in MODEL_CONFIGS
    #         for total_tokens in TRAIN_TOKEN_CONFIGS
    #     ]
    #     for result in run_training_job.starmap(
    #         [(cfg, tok, condition, test_run) for cfg, tok in jobs]
    #     ):
    #         print(f"Completed: {result}")

    #     run_fit_job.remote(condition, loss_type, test_run)