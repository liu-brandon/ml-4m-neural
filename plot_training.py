#!/usr/bin/env python3
"""
Plot training and Fixed Eval curves from 4M scaling experiment logs.

Auto-discovers all runs under a sweep directory and produces:
  training_loss.png    — epoch-level training loss per run
  fixed_eval.png       — Fixed Eval (CC12M + THINGS) vs tokens, all dims + conditions
  meg_rvq.png          — MEG RVQ layer losses with ln(512) chance baseline
  condition_gap.png    — loss(neural arm) - loss(control) vs tokens per dim

Usage:
  # Auto-discover all runs under a sweep directory
  python plot_training.py --sweep_dir /project/data/scaling_sweep --outdir /project/data/plots

  # Specific log files with labels:  path:label
  python plot_training.py \
    --logs /project/data/scaling_sweep/rgb_only/dim256_layer3_tok5.0B/checkpoints/log.txt:rgb_256 \
           /project/data/scaling_sweep/pixel_meg/dim256_layer3_tok5.0B/checkpoints/log.txt:meg_256 \
    --outdir ./plots
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─── colour / style conventions ─────────────────────────────────────────────

DIM_COLORS = {128: "#1f77b4", 192: "#ff7f0e", 256: "#2ca02c", 512: "#d62728"}
CONDITION_STYLES = {
    "rgb_only":                    "-",
    "rgb_only_pure_all2all":       "-.",
    "pixel_meg":                   "--",
    "pixel_meg_rvq0":              (0, (3, 1, 1, 1)),    # densely dashdotted
    "pixel_meg_shuffled":          (0, (5, 5)),           # loose dashes
    "pixel_meg_rvq0_shuffled":     (0, (3, 3, 1, 3)),    # dash-dot loose
    "pixel_meg_avg":               (0, (5, 1)),           # dense dashes
    "pixel_meg_avg_rvq0":          (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    "pixel_meg_avg_shuffled":      (0, (5, 1, 1, 1)),    # dense dash-dot
    "pixel_meg_avg_rvq0_shuffled": (0, (1, 1)),           # dotted
    "pixel_eeg":                   ":",
    "pixel_dinov2":                (0, (4, 1, 1, 1, 1, 1)),  # dash-dot-dot (positive control)
}
CONDITION_LABELS = {
    "rgb_only":                    "RGB-only (rgb→depth bias)",
    "rgb_only_pure_all2all":       "RGB-only (all2all)",
    "pixel_meg":                   "Pixel+MEG",
    "pixel_meg_rvq0":              "Pixel+MEG (rvq0 only)",
    "pixel_meg_shuffled":          "Pixel+MEG (shuffled)",
    "pixel_meg_rvq0_shuffled":     "Pixel+MEG rvq0 (shuffled)",
    "pixel_meg_avg":               "Pixel+MEG avg",
    "pixel_meg_avg_rvq0":          "Pixel+MEG avg (rvq0 only)",
    "pixel_meg_avg_shuffled":      "Pixel+MEG avg (shuffled)",
    "pixel_meg_avg_rvq0_shuffled": "Pixel+MEG avg rvq0 (shuffled)",
    "pixel_eeg":                   "Pixel+EEG",
    "pixel_dinov2":                "Pixel+DINOv2 (positive ctrl)",
}


def _legend(ax, **kw) -> None:
    """Call ax.legend only if there are labeled artists (avoids UserWarning)."""
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kw)

MEG_CHANCE = np.log(512)  # ln(512) ≈ 6.238 nats (uniform over 512-vocab codebook)


# ─── log parsing ────────────────────────────────────────────────────────────

def _parse_train_log_text(
    path: Path,
    epoch_size: int = 300_000,
    tokens_per_sample: int = 512,
) -> list[dict]:
    """Parse raw-text train.log (stdout format) into epoch dicts.

    Tracks current epoch from 'Epoch: [N]' lines, then reads
    'Averaged stats: key: val (epoch_avg) ...' lines as epoch summaries.
    Returns one dict per completed epoch with keys prefixed '[Epoch] '.
    Only training loss stats are present — no Fixed Eval data.
    """
    _epoch_re = re.compile(r"^Epoch: \[(\d+)\]")
    _avg_re = re.compile(r"^Averaged stats:\s+(.*)")
    _kv_re = re.compile(r"([\w@]+(?:@\d+)?(?:_[\w@]+)*(?:@\d+)?)\s*:\s*([\d.eE+\-]+)\s*\(([\d.eE+\-]+)\)")

    rows = []
    current_epoch = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _epoch_re.match(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            m = _avg_re.match(line)
            if m and current_epoch is not None:
                kvs = _kv_re.findall(m.group(1))
                if not kvs:
                    continue
                row: dict = {}
                for key, _, epoch_avg in kvs:
                    row[f"[Epoch] {key}"] = float(epoch_avg)
                tokens_b = (current_epoch + 1) * epoch_size * tokens_per_sample / 1e9
                row["total_tokens_seen_b"] = round(tokens_b, 6)
                row["epoch"] = current_epoch
                rows.append(row)

    return rows


def _normalize_json_row(data: dict) -> dict:
    """Prefix bare training-loss keys with '[Epoch] ' for consistency with text-log format.

    In checkpoints/log.txt, training losses are stored as plain keys like
    'tok_rgb@224_loss', while _parse_train_log_text produces '[Epoch] tok_rgb@224_loss'.
    Normalising here means all downstream plotting code can use one key convention.
    Fixed-Eval keys (starting with '[') are left unchanged.
    """
    result = {}
    for k, v in data.items():
        if k.endswith("_loss") and not k.startswith("["):
            result[f"[Epoch] {k}"] = v
        else:
            result[k] = v
    return result


def parse_log(path: Path) -> list[dict]:
    """Parse checkpoints/log.txt or train.log into a list of epoch dicts.

    Tries JSON first (checkpoints/log.txt format — has both training and Fixed Eval data).
    Falls back to text parser for train.log (training stats only, no eval).
    Only rows containing 'total_tokens_seen_b' are returned.
    JSON rows are normalised so bare training-loss keys gain the '[Epoch] ' prefix.
    """
    rows = []
    has_json = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                data = json.loads(line)
                if "total_tokens_seen_b" in data:
                    rows.append(_normalize_json_row(data))
                    has_json = True
            except json.JSONDecodeError:
                continue

    if has_json:
        return rows

    return _parse_train_log_text(path)


# ─── run discovery ──────────────────────────────────────────────────────────

def _dim_layers_from_name(run_name: str) -> tuple[int, int] | None:
    """Parse dim and layers from run directory name like 'dim512_layer3_tok5.0B'."""
    parts = run_name.split("_")
    dim = layers = None
    for p in parts:
        if p.startswith("dim"):
            try:
                dim = int(p[3:])
            except ValueError:
                pass
        if p.startswith("layer"):
            try:
                layers = int(p[5:])
            except ValueError:
                pass
    return (dim, layers) if dim and layers else None


def discover_runs(
    sweep_dir: Path,
    conditions: list[str] | None = None,
) -> dict[tuple[str, int, int], list[dict]]:
    """Walk sweep_dir and return {(condition, dim, layers): rows}.

    sweep_dir structure:
      sweep_dir/<condition>/<run_name>/checkpoints/log.txt
    """
    runs = {}
    for cond_dir in sorted(sweep_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        if conditions and cond not in conditions:
            continue
        for run_dir in sorted(cond_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            dl = _dim_layers_from_name(run_dir.name)
            if dl is None:
                continue
            dim, layers = dl
            log_path = run_dir / "checkpoints" / "log.txt"
            if not log_path.exists():
                # fall back to train.log in the run dir
                log_path = run_dir / "train.log"
            if not log_path.exists():
                continue
            rows = parse_log(log_path)
            if rows:
                runs[(cond, dim, layers)] = rows
                print(f"  loaded {len(rows)} epochs  {cond}/{run_dir.name}")
    return runs


# ─── helpers ────────────────────────────────────────────────────────────────

def _series(rows: list[dict], key: str) -> tuple[list, list]:
    """(x=tokens_b, y=value) for rows that contain key."""
    x, y = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            x.append(r["total_tokens_seen_b"])
            y.append(v)
    return x, y


def _label(cond: str, dim: int, layers: int) -> str:
    cname = CONDITION_LABELS.get(cond, cond)
    return f"{cname} dim{dim} L{layers}"


def _style(cond: str, dim: int):
    return dict(
        color=DIM_COLORS.get(dim, "gray"),
        linestyle=CONDITION_STYLES.get(cond, "-"),
    )


# ─── plot 1: training loss ───────────────────────────────────────────────────

def plot_training_loss(
    runs: dict[tuple, list[dict]],
    outdir: Path,
    keys: dict[str, str] | None = None,
):
    if keys is None:
        keys = {
            "Depth": "[Epoch] tok_depth@224_loss",
            "RGB":   "[Epoch] tok_rgb@224_loss",
        }
        # add MEG/DINOv2 columns if any run has them
        for (cond, dim, layers), rows in runs.items():
            if any("[Epoch] tok_meg_rvq0_loss" in r for r in rows):
                keys["MEG rvq0"] = "[Epoch] tok_meg_rvq0_loss"
            if any("[Epoch] tok_dinov2@224_loss" in r for r in rows):
                keys["DINOv2"] = "[Epoch] tok_dinov2@224_loss"

    ncols = len(keys)
    _, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, key) in zip(axes, keys.items()):
        for (cond, dim, layers), rows in sorted(runs.items()):
            x, y = _series(rows, key)
            if x:
                ax.plot(x, y, label=_label(cond, dim, layers), **_style(cond, dim), linewidth=1.5)
        ax.set_xlabel("Tokens seen (B)")
        ax.set_ylabel("Training loss (nats)")
        ax.set_title(f"Training — {name}")
        _legend(ax, fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = outdir / "training_loss.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─── plot 2: fixed eval ──────────────────────────────────────────────────────

def plot_fixed_eval(
    runs: dict[tuple, list[dict]],
    outdir: Path,
):
    panels = [
        ("CC12M Depth",  "[Fixed Eval (cc12m)] tok_depth@224_loss"),
        ("CC12M RGB",    "[Fixed Eval (cc12m)] tok_rgb@224_loss"),
        ("THINGS Depth", "[Fixed Eval (things)] tok_depth@224_loss"),
        ("THINGS RGB",   "[Fixed Eval (things)] tok_rgb@224_loss"),
    ]
    # include THINGS MEG rvq0 / DINOv2 panels if any run has them
    has_meg = any(
        any("[Fixed Eval (things)] tok_meg_rvq0_loss" in r for r in rows)
        for rows in runs.values()
    )
    if has_meg:
        panels.append(("THINGS MEG rvq0", "[Fixed Eval (things)] tok_meg_rvq0_loss"))

    has_dinov2 = any(
        any("[Fixed Eval (things)] tok_dinov2@224_loss" in r for r in rows)
        for rows in runs.values()
    )
    if has_dinov2:
        panels.append(("THINGS DINOv2", "[Fixed Eval (things)] tok_dinov2@224_loss"))

    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (name, key) in zip(axes_flat, panels):
        for (cond, dim, layers), rows in sorted(runs.items()):
            x, y = _series(rows, key)
            if x:
                ax.plot(x, y, "o-", label=_label(cond, dim, layers),
                        **_style(cond, dim), linewidth=1.5, markersize=4)
        ax.set_xlabel("Tokens seen (B)")
        ax.set_ylabel("Fixed Eval loss (nats)")
        ax.set_title(f"Fixed Eval — {name}")
        _legend(ax, fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = outdir / "fixed_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─── plot 3: MEG RVQ breakdown ───────────────────────────────────────────────

def _plot_meg_rvq_variant(
    runs: dict[tuple, list[dict]],
    outdir: Path,
    key_prefix: str,
    title_prefix: str,
    filename: str,
) -> None:
    eval_key_0 = f"[Fixed Eval (things)] {key_prefix}0_loss"
    active = {k: v for k, v in runs.items() if any(eval_key_0 in r for r in v)}
    if not active:
        return

    _, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, ax in enumerate(axes):
        key = f"[Fixed Eval (things)] {key_prefix}{i}_loss"
        for (cond, dim, layers), rows in sorted(active.items()):
            x, y = _series(rows, key)
            if x:
                ax.plot(x, y, "o-", label=_label(cond, dim, layers),
                        **_style(cond, dim), linewidth=1.5, markersize=4)
        ax.axhline(MEG_CHANCE, color="black", linestyle="--", linewidth=1,
                   label=f"chance  ln(512) = {MEG_CHANCE:.2f}")
        ax.set_xlabel("Tokens seen (B)")
        ax.set_ylabel("Fixed Eval loss (nats)")
        ax.set_title(f"{title_prefix} rvq{i}  (THINGS Fixed Eval)")
        _legend(ax, fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = outdir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_meg_rvq(runs: dict[tuple, list[dict]], outdir: Path):
    _plot_meg_rvq_variant(runs, outdir, "tok_meg_rvq", "MEG", "meg_rvq.png")
    _plot_meg_rvq_variant(runs, outdir, "tok_meg_avg_rvq", "MEG avg", "meg_avg_rvq.png")


# ─── plot 4: train vs eval combined ─────────────────────────────────────────

_TRAIN_EVAL_PANELS = [
    # (title,  train_key,                           eval_key)
    ("RGB (cc12m)",       "[Epoch] tok_rgb@224_loss",       "[Fixed Eval (cc12m)] tok_rgb@224_loss"),
    ("Depth (cc12m)",     "[Epoch] tok_depth@224_loss",     "[Fixed Eval (cc12m)] tok_depth@224_loss"),
    ("MEG rvq0",          "[Epoch] tok_meg_rvq0_loss",      "[Fixed Eval (things)] tok_meg_rvq0_loss"),
    ("MEG rvq1",          "[Epoch] tok_meg_rvq1_loss",      "[Fixed Eval (things)] tok_meg_rvq1_loss"),
    ("MEG rvq2",          "[Epoch] tok_meg_rvq2_loss",      "[Fixed Eval (things)] tok_meg_rvq2_loss"),
    ("MEG rvq3",          "[Epoch] tok_meg_rvq3_loss",      "[Fixed Eval (things)] tok_meg_rvq3_loss"),
    ("MEG avg rvq0",      "[Epoch] tok_meg_avg_rvq0_loss",  "[Fixed Eval (things)] tok_meg_avg_rvq0_loss"),
    ("MEG avg rvq1",      "[Epoch] tok_meg_avg_rvq1_loss",  "[Fixed Eval (things)] tok_meg_avg_rvq1_loss"),
    ("MEG avg rvq2",      "[Epoch] tok_meg_avg_rvq2_loss",  "[Fixed Eval (things)] tok_meg_avg_rvq2_loss"),
    ("MEG avg rvq3",      "[Epoch] tok_meg_avg_rvq3_loss",  "[Fixed Eval (things)] tok_meg_avg_rvq3_loss"),
    ("EEG",               "[Epoch] tok_eeg_loss",           "[Fixed Eval (things)] tok_eeg_loss"),
    ("DINOv2",            "[Epoch] tok_dinov2@224_loss",    "[Fixed Eval (things)] tok_dinov2@224_loss"),
]


def plot_train_vs_eval(runs: dict[tuple, list[dict]], outdir: Path):
    """Training loss + Fixed Eval loss on the same axes per modality.

    Faded lines = per-epoch training loss (dense); solid markers = Fixed Eval
    on the held-out set (sparse, logged every eval_freq epochs).  Plotting both
    together makes train/val divergence immediately visible.
    Only panels where at least one run has data for both keys are shown.
    """
    panels = [
        (title, tkey, ekey)
        for title, tkey, ekey in _TRAIN_EVAL_PANELS
        if any(
            any(tkey in r for r in rows) or any(ekey in r for r in rows)
            for rows in runs.values()
        )
    ]
    if not panels:
        return

    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for ax, (title, train_key, eval_key) in zip(axes_flat, panels):
        for (cond, dim, layers), rows in sorted(runs.items()):
            style = _style(cond, dim)
            label = _label(cond, dim, layers)

            tx, ty = _series(rows, train_key)
            if tx:
                ax.plot(tx, ty, alpha=0.35, linewidth=0.9, **style)

            ex, ey = _series(rows, eval_key)
            if ex:
                ax.plot(ex, ey, "o-", linewidth=1.5, markersize=4,
                        label=label, **style)

        ax.set_xlabel("Tokens seen (B)")
        ax.set_ylabel("Loss (nats)")
        ax.set_title(title)
        _legend(ax, fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    # Figure-level proxy legend explaining line styles
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], color="gray", linewidth=0.9, alpha=0.35, label="train (per epoch)"),
        Line2D([0], [0], color="gray", linewidth=1.5, marker="o", markersize=4, label="eval (fixed set)"),
    ]
    fig.legend(handles=proxy, loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = outdir / "train_vs_eval.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─── plot 5: condition gap (treatment − control) ─────────────────────────────

def plot_condition_gap(
    runs: dict[tuple, list[dict]],
    outdir: Path,
    control: str = "rgb_only",
    treatment: str | None = None,
    gap_keys: dict[str, str] | None = None,
):
    if gap_keys is None:
        gap_keys = {
            "CC12M Depth gap": "[Fixed Eval (cc12m)] tok_depth@224_loss",
            "CC12M RGB gap":   "[Fixed Eval (cc12m)] tok_rgb@224_loss",
        }

    all_conditions = {cond for (cond, _, _) in runs}
    if treatment is not None:
        treatments = [treatment]
    else:
        treatments = sorted(all_conditions - {control})

    if not treatments:
        return

    for treat in treatments:
        dims = sorted({dim for (cond, dim, _) in runs if cond in (control, treat)})
        if not dims:
            continue

        ncols = len(gap_keys)
        _, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
        axes = axes[0]

        for ax, (name, key) in zip(axes, gap_keys.items()):
            for dim in dims:
                ctrl_key  = next(((c, d, l) for c, d, l in runs if c == control and d == dim), None)
                treat_key = next(((c, d, l) for c, d, l in runs if c == treat   and d == dim), None)
                if ctrl_key is None or treat_key is None:
                    continue

                cx, cy = _series(runs[ctrl_key],  key)
                tx, ty = _series(runs[treat_key], key)

                if not cx or not tx:
                    continue
                cy_interp = np.interp(tx, cx, cy)
                gap = [t - c for t, c in zip(ty, cy_interp)]

                color = DIM_COLORS.get(dim, "gray")
                ax.plot(tx, gap, "o-", color=color, linewidth=1.5, markersize=4,
                        label=f"dim{dim}")

            ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
            ax.set_xlabel("Tokens seen (B)")
            ax.set_ylabel(
                f"{CONDITION_LABELS.get(treat, treat)} − {CONDITION_LABELS.get(control, control)}  (nats)"
            )
            ax.set_title(name)
            _legend(ax, fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        out = outdir / f"condition_gap_{treat}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep_dir", type=Path, default=None,
        help="Root of scaling sweep (contains condition subdirs). "
             "Mutually exclusive with --logs.",
    )
    parser.add_argument(
        "--logs", nargs="+", default=None,
        metavar="PATH:LABEL",
        help="Explicit log files as 'path:label' pairs.",
    )
    parser.add_argument(
        "--conditions", type=str, default=None,
        help="Comma-separated conditions to include, e.g. rgb_only,pixel_meg",
    )
    parser.add_argument(
        "--outdir", type=Path, default=None,
        help="Output directory for plots. When --sweep_dir is used and this is omitted, "
             "per-condition plots go to {sweep_dir}/{condition}/plots/ and comparison "
             "plots go to {sweep_dir}/plots/.",
    )
    parser.add_argument(
        "--control", type=str, default="rgb_only",
        help="Condition to use as control for gap plot",
    )
    parser.add_argument(
        "--treatment", type=str, default=None,
        help="Condition to use as treatment for gap plot. "
             "Omit to auto-generate gap plots for all non-control conditions.",
    )
    args = parser.parse_args()

    conditions = [c for c in args.conditions.split(",") if c] if args.conditions else None

    if args.sweep_dir:
        sweep_dir = args.sweep_dir
        print(f"Discovering runs under {sweep_dir} ...")
        all_runs = discover_runs(sweep_dir, conditions=conditions)

        if not all_runs:
            print("No runs found.")
            return

        # Per-condition individual plots
        conds_in_runs = sorted({cond for (cond, _, _) in all_runs})
        for cond in conds_in_runs:
            cond_runs = {k: v for k, v in all_runs.items() if k[0] == cond}
            outdir = args.outdir or (sweep_dir / cond / "plots")
            outdir.mkdir(parents=True, exist_ok=True)
            print(f"\nPlotting {cond} ({len(cond_runs)} runs) → {outdir}")
            plot_training_loss(cond_runs, outdir)
            plot_fixed_eval(cond_runs, outdir)
            plot_meg_rvq(cond_runs, outdir)
            plot_train_vs_eval(cond_runs, outdir)

        # Comparison plots across all discovered conditions
        comparison_outdir = args.outdir or (sweep_dir / "plots")
        comparison_outdir.mkdir(parents=True, exist_ok=True)
        print(f"\nComparison plots ({len(all_runs)} runs) → {comparison_outdir}")
        plot_training_loss(all_runs, comparison_outdir)
        plot_fixed_eval(all_runs, comparison_outdir)
        plot_train_vs_eval(all_runs, comparison_outdir)
        plot_condition_gap(all_runs, comparison_outdir,
                           control=args.control, treatment=args.treatment)

    elif args.logs:
        # Manual mode: parse "path:label" pairs; label used as (cond, dim, layers) fallback
        runs = {}
        for entry in args.logs:
            path_str, _, label = entry.partition(":")
            path = Path(path_str)
            dl = _dim_layers_from_name(label) or (0, 0)
            key = (label, dl[0], dl[1])
            rows = parse_log(path)
            if rows:
                runs[key] = rows
                print(f"  loaded {len(rows)} epochs  {label}")
        if not runs:
            print("No runs found.")
            return
        outdir = args.outdir or Path("./plots")
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\nPlotting {len(runs)} runs → {outdir}")
        plot_training_loss(runs, outdir)
        plot_fixed_eval(runs, outdir)
        plot_meg_rvq(runs, outdir)
        plot_train_vs_eval(runs, outdir)
        plot_condition_gap(runs, outdir, control=args.control, treatment=args.treatment)

    else:
        parser.error("Provide either --sweep_dir or --logs.")

    print("Done.")


if __name__ == "__main__":
    main()
