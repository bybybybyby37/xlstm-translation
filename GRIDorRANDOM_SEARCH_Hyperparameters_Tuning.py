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


# Grid Search
# python GRIDorRANDOM_SEARCH_Hyperparameters_Tuning.py --mode grid

# Random Search，for 10 combi
# python GRIDorRANDOM_SEARCH_Hyperparameters_Tuning.py --mode random --n-trials 10

# ==========================
# Basic configuration
# ==========================

BASE_CONFIG = "config/iwslt17_xlstm10.yaml"
VARIANT = "10"
EXPERIMENT_ROOT = "grid_runs/xlstm10"


# ==========================
# Utilities
# ==========================

def ensure_experiment_root():
    """Create experiment root directory if it does not exist."""
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)
    print("Experiment root:", os.path.abspath(EXPERIMENT_ROOT))


def set_nested(cfg, dotted_key, value):
    """
    Set a value in a nested OmegaConf config using a dotted key, e.g. "training.lr".
    """
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_metrics(stdout: str):
    """
    Parse test_loss, test_ppl, and test_bleu from the training script stdout.

    Expected patterns (based on your train_iwslt17_xlstm.py):
      - "Test loss: {test_loss:.4f}, ppl={math.exp(test_loss):.2f}"
      - "Test BLEU on {N} samples: {bleu.score:.2f}"
    """
    test_loss = None
    test_ppl = None
    test_bleu = None

    for line in stdout.splitlines():
        line = line.strip()

        # Parse test loss and ppl
        if line.startswith("Test loss:"):
            m = re.search(r"Test loss:\s*([0-9.]+),\s*ppl=([0-9.]+)", line)
            if m:
                test_loss = float(m.group(1))
                test_ppl = float(m.group(2))

        # Parse test BLEU
        if line.startswith("Test BLEU on"):
            m = re.search(r"Test BLEU on \d+ samples:\s*([0-9.]+)", line)
            if m:
                test_bleu = float(m.group(1))

    return test_loss, test_ppl, test_bleu


def build_run_name(config_overrides: dict):
    """
    Build a unique run name based on the overridden hyperparameters and timestamp.
    """
    suffix_parts = []
    for k, v in sorted(config_overrides.items(), key=lambda x: x[0]):
        key_str = k.replace(".", "-")
        if isinstance(v, float):
            # Use a compact scientific notation for floats
            v_str = f"{v:.0e}"
        else:
            v_str = str(v)
        suffix_parts.append(f"{key_str}={v_str}")

    suffix = "_".join(suffix_parts)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"xlstm10_{suffix}_{ts}"
    return run_name


def run_one(config_overrides: dict):
    """
    Run a single training job with given config overrides.

    Steps:
      1. Load base config.
      2. Apply overrides.
      3. Save to a new YAML file.
      4. Call the training script via subprocess.
      5. Save logs.
      6. Parse metrics from stdout.
    """
    # 1. Load base config
    cfg = OmegaConf.load(BASE_CONFIG)

    # 2. Apply overrides
    for k, v in config_overrides.items():
        set_nested(cfg, k, v)

    # 3. Save config
    run_name = build_run_name(config_overrides)
    cfg_path = os.path.join(EXPERIMENT_ROOT, run_name + ".yaml")
    OmegaConf.save(cfg, cfg_path)

    print(f"\n=== Running {run_name} ===")
    print("Config overrides:", config_overrides)
    print("Config path:", cfg_path)

    # 4. Build command
    cmd = [
        "python",
        "-m",
        "scripts.train_iwslt17_xlstm",
        "--config",
        cfg_path,
        "--variant",
        VARIANT,
    ]
    print("Command:", " ".join(cmd))

    # 5. Run training
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 6. Save logs
    log_path = os.path.join(EXPERIMENT_ROOT, run_name + "_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(result.stderr)

    print("Return code:", result.returncode)
    print("Log saved to:", log_path)

    # 7. Parse metrics
    test_loss, test_ppl, test_bleu = parse_metrics(result.stdout)

    return {
        "run_name": run_name,
        "overrides": config_overrides,
        "config_path": cfg_path,
        "log_path": log_path,
        "returncode": result.returncode,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "test_bleu": test_bleu,
    }


def save_results(results, filename="search_results.csv"):
    """
    Convert results list to a DataFrame, save as CSV, and print a short summary.
    """
    df_rows = []
    for r in results:
        row = {
            "run_name": r["run_name"],
            "returncode": r["returncode"],
            "test_loss": r["test_loss"],
            "test_ppl": r["test_ppl"],
            "test_bleu": r["test_bleu"],
        }
        for k, v in r["overrides"].items():
            row[k] = v
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    # Try to sort by BLEU if available, otherwise by loss
    sort_cols = []
    ascending = []
    if "test_bleu" in df.columns:
        sort_cols.append("test_bleu")
        ascending.append(False)
    if "test_loss" in df.columns:
        sort_cols.append("test_loss")
        ascending.append(True)

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=ascending)

    out_path = os.path.join(EXPERIMENT_ROOT, filename)
    df.to_csv(out_path, index=False)

    print("\n===== Search summary =====")
    print(df)
    print("\nSaved results to:", out_path)

    return df


# ==========================
# Grid Search
# ==========================

def run_grid_search():
    """
    Perform a simple grid search over a manually defined param_grid.

    You can modify param_grid to suit your needs.
    """
    # Define your grid here
    param_grid = {
        "training.lr": [1e-3, 5e-4, 1e-4],
        "training.batch_size": [16, 32],
        # "model.num_blocks": [4, 6],
        # "model.embedding_dim": [256, 384],
    }

    keys, values_list = zip(*param_grid.items())
    results = []

    print("\n=== Starting GRID SEARCH ===")
    print("param_grid:", param_grid)

    for combo in itertools.product(*values_list):
        overrides = dict(zip(keys, combo))
        res = run_one(overrides)
        results.append(res)

    df = save_results(results, filename="grid_search_results.csv")
    return df


# ==========================
# Random Search
# ==========================

def sample_from_space(space_def):
    """
    Sample a single value from a search space definition.

    Supported format examples:
      - {"type": "choice", "values": [16, 32, 64]}
      - {"type": "loguniform", "low": 1e-4, "high": 5e-4}
      - {"type": "uniform", "low": 0.0, "high": 1.0}
    """
    s_type = space_def["type"]

    if s_type == "choice":
        values = space_def["values"]
        return random.choice(values)

    elif s_type == "loguniform":
        low = space_def["low"]
        high = space_def["high"]
        # Sample in log space
        log_low = math.log(low)
        log_high = math.log(high)
        sample_log = random.uniform(log_low, log_high)
        return math.exp(sample_log)

    elif s_type == "uniform":
        low = space_def["low"]
        high = space_def["high"]
        return random.uniform(low, high)

    else:
        raise ValueError(f"Unsupported search space type: {s_type}")


def run_random_search(n_trials: int):
    """
    Perform random search over a defined search space for n_trials runs.
    """
    # Define your random search space here
    search_space = {
        "training.lr": {
            "type": "loguniform",
            "low": 1e-4,
            "high": 5e-4,
        },
        "training.batch_size": {
            "type": "choice",
            "values": [16, 32, 64],
        },
        "model.num_blocks": {
            "type": "choice",
            "values": [4, 6],
        },
        # "model.embedding_dim": {
        #     "type": "choice",
        #     "values": [256, 384],
        # },
    }

    results = []

    print("\n=== Starting RANDOM SEARCH ===")
    print("search_space:", search_space)
    print("n_trials:", n_trials)

    for i in range(n_trials):
        overrides = {}
        for key, sp_def in search_space.items():
            overrides[key] = sample_from_space(sp_def)

        print(f"\n--- Random trial {i + 1}/{n_trials} ---")
        print("Sampled overrides:", overrides)

        res = run_one(overrides)
        results.append(res)

    df = save_results(results, filename="random_search_results.csv")
    return df


# ==========================
# Main entry
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid / Random Search Hyperparameter Tuning for xLSTM-10 IWSLT17"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["grid", "random"],
        default="grid",
        help="Search mode: 'grid' or 'random'.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials for random search (ignored in grid mode).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_experiment_root()

    if args.mode == "grid":
        run_grid_search()
    else:
        run_random_search(args.n_trials)


if __name__ == "__main__":
    main()
