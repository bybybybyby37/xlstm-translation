import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# ---- user-editable: define the 3-way comparison here ----
RUNS = [
    # ("01", "config/iwslt17_xlstm01.yaml"),
    # ("10", "config/iwslt17_xlstm10.yaml"),
    ("11", "config/iwslt17_xlstm11.yaml"),
]

EVAL_SPLIT = "test"  # "val" or "test"
LOG_DIR = Path("logs") / f"iwslt17_xlstm_3way_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Optional: pin GPU(s) for reproducibility / avoiding accidental multi-GPU changes
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_one(variant: str, cfg_path: str) -> int:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = LOG_DIR / f"train_variant_{variant}_{ts}.log"

    # snapshot the yaml to make the run reproducible for your paper
    cfg_src = Path(cfg_path)
    cfg_dst = LOG_DIR / f"{cfg_src.stem}_snapshot_{variant}.yaml"
    shutil.copy2(cfg_src, cfg_dst)

    cmd = [
        sys.executable, "-m", "scripts.train_iwslt17_xlstm",
        "--config", str(cfg_dst),      # use the snapshot
        "--variant", variant,
        "--eval_split", EVAL_SPLIT,    # explicit
    ]

    print("=" * 80)
    print(f"[RUN] variant={variant}")
    print("Command:", " ".join(cmd))
    print(f"Log file: {log_file}")
    print(f"Config snapshot: {cfg_dst}")
    print("=" * 80)

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        return process.wait()

def main():
    for variant, cfg in RUNS:
        ret = run_one(variant, cfg)
        if ret != 0:
            print(f"[FAIL] variant {variant} exited with code {ret}. Check logs under: {LOG_DIR}")
            sys.exit(ret)
        print(f"[OK] variant {variant} finished.")

    print(f"[DONE] all runs finished. Logs & config snapshots are under: {LOG_DIR}")

if __name__ == "__main__":
    main()
