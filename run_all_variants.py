import subprocess
import sys
from datetime import datetime

runs = [
    ("10", "config/iwslt17_xlstm10.yaml"),
    ("01", "config/iwslt17_xlstm01.yaml"),
    ("11", "config/iwslt17_xlstm11.yaml"),
]

for variant, cfg in runs:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"train_variant_{variant}_{ts}.log"

    cmd = [
        sys.executable, "-m", "scripts.train_iwslt17_xlstm",
        "--config", cfg,
        "--variant", variant,
    ]

    print("=" * 80)
    print(f"Starting variant {variant}")
    print("Command:", " ".join(cmd))
    print(f"Logging to {log_file}")
    print("=" * 80)

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        ret = process.wait()

    if ret != 0:
        print(f"Variant {variant} FAILED (exit code {ret}). Check log: {log_file}")
        break

    print(f"Variant {variant} finished successfully!")

print("All scheduled runs complete.")
