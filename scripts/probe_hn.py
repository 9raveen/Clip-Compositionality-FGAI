"""Hard-negative probe: how often does each model rank a caption above its negative, per negative type?

For every validation pair (caption, negative) on the same image:  margin = cos(img, caption) - cos(img, negative);
hn_acc = fraction with margin > 0.  Evaluated for zero-shot, B, C and C-plaus on BOTH validation files
(all validated negatives, and the likelihood-matched subset) so conditions are compared on identical pairs.
Reading it: if C gains a lot on the shortcut-prone types (rel_replace, obj_swap) but little elsewhere, and
Winoground does not move, the gain is consistent with text-only shortcuts. That is evidence, not proof.

  python -m scripts.probe_hn                                   # zeroshot, lora_baseline, lora_hn, lora_hn_plaus x seeds 0-2
  python -m scripts.probe_hn --conditions zeroshot lora_baseline lora_hn_attr --seeds 0
"""
import argparse, collections, json, os
import numpy as np

TYPES = ["attr_replace", "attr_swap", "obj_replace", "obj_swap", "rel_replace"]


def pair_margins(img_emb, txt_emb, rows):
    """Pure function (unit-tested). img_emb: {image_id: unit vec}, txt_emb: {text: unit vec}.
    Returns {type: [margin, ...]} over every (caption, negative) pair."""
    out = collections.defaultdict(list)
    for r in rows:
        v, c = img_emb[r["image_id"]], txt_emb[r["caption"]]
        for n in r["negatives"]:
            out[n["type"]].append(float(v @ c - v @ txt_emb[n["text"]]))
    return out


def summarize(margins):
    return {t: {"n": len(m), "hn_acc": float(np.mean(np.array(m) > 0)), "mean_margin": float(np.mean(m))}
            for t, m in margins.items()}


def embed_all(model, processor, device, rows, image_dir, batch=128):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.model import feat
    ids = sorted({r["image_id"] for r in rows})
    texts = sorted({r["caption"] for r in rows} | {n["text"] for r in rows for n in r["negatives"]})
    img_emb, txt_emb = {}, {}
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            chunk = ids[s:s + batch]
            ims = [Image.open(os.path.join(image_dir, f"{i:012d}.jpg")).convert("RGB") for i in chunk]
            px = processor(images=ims, return_tensors="pt")["pixel_values"].to(device)
            e = F.normalize(feat(model.get_image_features(pixel_values=px)).float(), dim=-1).cpu().numpy()
            img_emb.update(zip(chunk, e))
        for s in range(0, len(texts), batch):
            chunk = texts[s:s + batch]
            t = processor.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            e = F.normalize(feat(model.get_text_features(**t)).float(), dim=-1).cpu().numpy()
            txt_emb.update(zip(chunk, e))
    return img_emb, txt_emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["zeroshot", "lora_baseline", "lora_hn", "lora_hn_plaus"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--val_files", nargs="+", default=["data/negatives_val.jsonl", "data/negatives_val_plaus.jsonl"])
    ap.add_argument("--out_dir", default=os.environ.get("PROJECT_DIR", "outputs"))
    ap.add_argument("--out", default="results/probe_hn.json")
    args = ap.parse_args()
    import torch
    from transformers import CLIPProcessor
    from src.model import build_model
    from src.utils import load_config, load_trainable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = {f: [json.loads(l) for l in open(f)] for f in args.val_files}
    results = collections.defaultdict(lambda: collections.defaultdict(list))   # cond -> file -> [per-seed summary]
    processor = None
    for cond in args.conditions:
        cfg = load_config(f"configs/{cond}.yaml")
        processor = processor or CLIPProcessor.from_pretrained(cfg["model"])
        for seed in ([0] if cond == "zeroshot" else args.seeds):
            model = build_model(cfg, device)
            if cond != "zeroshot":
                ck = os.path.join(args.out_dir, "ckpt", f"{cond}_seed{seed}_last.pt")
                if not os.path.exists(ck):
                    print(f"[skip] missing {ck}"); continue
                load_trainable(model, torch.load(ck, map_location=device)["trainable"])
            model.eval()
            for f, rows in files.items():
                img_emb, txt_emb = embed_all(model, processor, device, rows, cfg["image_dir"])
                results[cond][f].append({"seed": seed, **summarize(pair_margins(img_emb, txt_emb, rows))})
            del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    for f in files:
        print(f"\n=== hn_acc (%) per negative type on {f}  (mean ± std over seeds; n = pairs) ===")
        print(f"{'condition':16s} " + " ".join(f"{t:>16s}" for t in TYPES))
        for cond in args.conditions:
            runs = results[cond][f]
            if not runs:
                continue
            cells = []
            for t in TYPES:
                v = [100 * r[t]["hn_acc"] for r in runs if t in r]
                n = next((r[t]["n"] for r in runs if t in r), 0)
                cells.append((f"{np.mean(v):.1f}" + (f"±{np.std(v, ddof=1):.1f}" if len(v) > 1 else "") + f" (n={n})") if v else "-")
            print(f"{cond:16s} " + " ".join(f"{c:>16s}" for c in cells))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
