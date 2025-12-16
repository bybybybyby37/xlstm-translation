"""
============================================================
Grid / Random Search for xLSTM (IWSLT17 En-Zh)

USAGE:

# Grid Search
python GRIDorRANDOM_SEARCH_Hyperparameters_Tuning.py \
    --mode grid \
    --variant 11

# Random Search (recommended)
python GRIDorRANDOM_SEARCH_Hyperparameters_Tuning.py \
    --mode random \
    --n-trials 8 \
    --variant 11

NOTES:
- SEARCH uses VALIDATION split only
- SEARCH forces short training budget:
    * num_epochs = 6
    * patience_epochs = 1
- Selection is based on eval_loss (token-average val loss)
- Each run has a UNIQUE checkpoint name (no overwrite)
============================================================
"""

import os
import itertools
import subprocess
import re
import argparse
import math
import random
from datetime import datetime

from omegaconf import OmegaConf
import pandas as pd


# ======================================================
# Basic configuration (can be overridden by CLI)
# ======================================================

BASE_CONFIG = "config/iwslt17_xlstm11.yaml"
EXPERIMENT_ROOT = "search_runs"


# ======================================================
# Utilities
# ======================================================

def ensure_experiment_root(root):
    os.makedirs(root, exist_ok=True)
    print("Experiment root:", os.path.abspath(root))


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_metrics(stdout: str):
    """
    Parse evaluation metrics from training stdout.
    Compatible with both VAL / TEST outputs.
    """
    eval_loss = None
    eval_ppl = None
    eval_bleu = None

    for line in stdout.splitlines():
        line = line.strip()

        if line.startswith(("VAL loss:", "TEST loss:")):
            m = re.search(r"loss:\s*([0-9.]+),\s*ppl=([0-9.]+)", line)
            if m:
                eval_loss = float(m.group(1))
                eval_ppl = float(m.group(2))

        if line.startswith(("VAL BLEU", "TEST BLEU")):
            m = re.search(r"BLEU.*?:\s*([0-9.]+)", line)
            if m:
                eval_bleu = float(m.group(1))

    return eval_loss, eval_ppl, eval_bleu


def build_run_name(variant: str, overrides: dict):
    parts = []
    for k, v in sorted(overrides.items()):
        key = k.replace(".", "-")
        if isinstance(v, float):
            v = f"{v:.1e}"
        parts.append(f"{key}={v}")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"xlstm{variant}_" + "_".join(parts) + "_" + ts


# ======================================================
# Single run
# ======================================================

def run_one(base_cfg_path: str, variant: str, root: str, overrides: dict):
    cfg = OmegaConf.load(base_cfg_path)

    # ----- apply hyperparameter overrides -----
    for k, v in overrides.items():
        set_nested(cfg, k, v)

    # ----- force SEARCH budget (critical) -----
    set_nested(cfg, "training.num_epochs", 6)
    set_nested(cfg, "training.patience_epochs", 1)

    run_name = build_run_name(variant, overrides)

    cfg_path = os.path.join(root, run_name + ".yaml")
    OmegaConf.save(cfg, cfg_path)

    print(f"\n=== Running {run_name} ===")
    print("Overrides:", overrides)
    print("Config:", cfg_path)

    cmd = [
        "python",
        "-m",
        "scripts.train_iwslt17_xlstm",
        "--config", cfg_path,
        "--variant", run_name,   # UNIQUE ckpt name
        "--eval_split", "val",   # SEARCH uses validation only
    ]

    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    log_path = os.path.join(root, run_name + "_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)

    eval_loss, eval_ppl, eval_bleu = parse_metrics(result.stdout)

    return {
        "run_name": run_name,
        "returncode": result.returncode,
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "eval_bleu": eval_bleu,
        **overrides,
    }


# ======================================================
# Grid Search
# ======================================================

def run_grid_search(args, root):
    param_grid = {
        "training.lr": [5e-4, 3e-4, 1e-4],
        "training.weight_decay": [1e-2, 1e-3, 1e-4],
    }

    keys, values = zip(*param_grid.items())
    results = []

    print("\n=== GRID SEARCH ===")
    print(param_grid)

    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        results.append(run_one(args.base_config, args.variant, root, overrides))

    return save_results(results, root, "grid_search_results.csv")


# ======================================================
# Random Search
# ======================================================

def sample_from_space(space):
    if space["type"] == "choice":
        return random.choice(space["values"])
    if space["type"] == "loguniform":
        lo, hi = math.log(space["low"]), math.log(space["high"])
        return math.exp(random.uniform(lo, hi))
    raise ValueError(space)


def run_random_search(args, root):
    search_space = {
        "training.lr": {
            "type": "loguniform",
            "low": 3e-4,
            "high": 1.5e-3,
        },
        "training.weight_decay": {
            "type": "loguniform",
            "low": 1e-4,
            "high": 2e-2,
        },
    }

    results = []

    print("\n=== RANDOM SEARCH ===")
    print("Trials:", args.n_trials)
    print(search_space)

    for i in range(args.n_trials):
        overrides = {k: sample_from_space(v) for k, v in search_space.items()}
        print(f"\n--- Trial {i+1}/{args.n_trials} ---")
        results.append(run_one(args.base_config, args.variant, root, overrides))

    return save_results(results, root, "random_search_results.csv")


# ======================================================
# Save & summarize
# ======================================================

def save_results(results, root, filename):
    df = pd.DataFrame(results)

    df = df.sort_values(
        by=["eval_loss"],
        ascending=[True],
    )

    out_path = os.path.join(root, filename)
    df.to_csv(out_path, index=False)

    print("\n===== SEARCH SUMMARY =====")
    print(df)
    print("\nSaved to:", out_path)

    return df


# ======================================================
# Main
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["grid", "random"], default="random")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument(
        "--variant",
        type=str,
        default="11",
        help="Model variant tag: 10 / 01 / 11 (used in run name & ckpt)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=BASE_CONFIG,
        help="Base YAML config path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = os.path.join(EXPERIMENT_ROOT, f"xlstm{args.variant}")
    ensure_experiment_root(root)

    if args.mode == "grid":
        run_grid_search(args, root)
    else:
        run_random_search(args, root)


if __name__ == "__main__":
    main()
