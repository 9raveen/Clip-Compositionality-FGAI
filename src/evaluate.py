"""Shared evaluation protocol: identical for zero-shot, baseline and hard-negative models.

  python -m src.evaluate --config configs/zeroshot.yaml                  # zero-shot, no training
  python -m src.evaluate --config configs/lora_hn.yaml --ckpt <adapter .pt>
"""
import argparse, json, os
import torch
from transformers import CLIPProcessor

from src.eval_winoground import bootstrap_ci, load_winoground, winoground_scores
from src.eval_retrieval import retrieval_recall
from src.model import build_model
from src.utils import git_hash, load_config, load_trainable


def run_eval(model, processor, device, cfg, wino=None):
    res = {"winoground": winoground_scores(model, processor, device, ds=wino, return_per_example=True)}
    for k in ("text", "image", "group"):
        res["winoground"][f"{k}_ci95"] = list(bootstrap_ci(res["winoground"]["per_example"][k]))
    if os.path.exists(cfg["test_jsonl"]):
        res["retrieval"] = retrieval_recall(model, processor, device, cfg["test_jsonl"], cfg["image_dir"])
    # TODO(C): ARO (VG-Relation, VG-Attribution) and optional SugarCrepe; add results under res["aro"], res["sugarcrepe"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_dir", default=os.environ.get("PROJECT_DIR", "outputs"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device)
    if args.ckpt:
        load_trainable(model, torch.load(args.ckpt, map_location=device)["trainable"])
    processor = CLIPProcessor.from_pretrained(cfg["model"])
    res = run_eval(model, processor, device, cfg)
    path = os.path.join(args.out_dir, "results", f"{cfg['name']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"config": cfg["name"], "git": git_hash(), **res}, open(path, "w"), indent=2)
    print({k: round(v, 4) for k, v in res["winoground"].items() if k in ("text", "image", "group")})
    print(res.get("retrieval"))
    print("saved", path)
