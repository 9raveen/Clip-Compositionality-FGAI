"""Run every (config, seed) pair, skipping finished ones. Safe to re-run after a Colab restart."""
import os, subprocess, sys

OUT = os.environ.get("PROJECT_DIR", "outputs")
CONFIGS = ["baseline_ft", "full_all_negs"]   # add ablation configs here
SEEDS = [0, 1, 2]

for cfg in CONFIGS:
    for seed in SEEDS:
        if os.path.exists(f"{OUT}/results/{cfg}_seed{seed}.json"):
            continue
        r = subprocess.run([sys.executable, "-m", "src.train", "--config", f"configs/{cfg}.yaml",
                            "--seed", str(seed), "--out_dir", OUT])
        if r.returncode != 0:
            print(f"[fail] {cfg} seed {seed}"); sys.exit(1)
