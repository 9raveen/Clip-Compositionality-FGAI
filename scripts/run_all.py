"""Run (config, seed) pairs by tier, skipping finished runs. Safe to re-run after a Colab restart.

  python scripts/run_all.py --tier main        # B and C x 3 seeds (the core comparison)
  python scripts/run_all.py --tier ablation    # category-only and lambda variants, seed 0
  python scripts/run_all.py --tier optional    # full fine-tuning, text-tower-only LoRA, seed 0
"""
import argparse, os, subprocess, sys

TIERS = {
    "main": (["lora_baseline", "lora_hn"], [0, 1, 2]),
    "ablation": (["lora_hn_attr", "lora_hn_obj", "lora_hn_rel", "lora_hn_lam05", "lora_hn_lam2"], [0]),
    "optional": (["full_hn", "lora_hn_textonly"], [0]),
}
ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=list(TIERS), required=True)
args = ap.parse_args()
OUT = os.environ.get("PROJECT_DIR", "outputs")
configs, seeds = TIERS[args.tier]
for seed in seeds:                       # seed-major order: a complete B/C pair per seed finishes first
    for cfg in configs:
        if os.path.exists(f"{OUT}/results/{cfg}_seed{seed}.json"):
            continue
        r = subprocess.run([sys.executable, "-m", "src.train", "--config", f"configs/{cfg}.yaml",
                            "--seed", str(seed), "--out_dir", OUT])
        if r.returncode != 0:
            print(f"[fail] {cfg} seed {seed}"); sys.exit(1)
