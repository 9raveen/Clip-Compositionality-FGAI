"""SugarCrepe evaluation (artifact-controlled compositional benchmark, built on COCO val2017 images).

For each example the model must score the true caption above a hard-negative caption for the same image.
We trained only on COCO train2017, so SugarCrepe images are unseen. SugarCrepe was built to remove the text-only
artifacts of older benchmarks, so it is the fair test of whether training on our rule-based negatives transferred.

Setup on Colab (once):
  !git clone -q https://github.com/RAIVNLab/sugar-crepe.git /content/sugar-crepe
  !wget -q http://images.cocodataset.org/zips/val2017.zip -P /content && unzip -q /content/val2017.zip -d /content/coco
Run:
  python -m scripts.eval_sugarcrepe --data_root /content/sugar-crepe/data --coco_image_root /content/coco/val2017
Files expected in data_root: replace_obj/att/rel.json, swap_obj/att.json, add_obj/att.json, each entries with
fields filename, caption, negative_caption (dict-of-entries or list). The script stops with a clear message otherwise.
"""
import argparse, collections, json, os
import numpy as np

TASKS = ["replace_obj", "replace_att", "replace_rel", "swap_obj", "swap_att", "add_obj", "add_att"]
FIELDS = {"filename", "caption", "negative_caption"}
PAIRS = [("lora_baseline", "lora_hn"), ("lora_baseline", "lora_hn_plaus"), ("lora_hn_plaus", "lora_hn")]


def load_task(data_root, task):
    d = json.load(open(os.path.join(data_root, f"{task}.json")))
    items = list(d.values()) if isinstance(d, dict) else d
    for it in items:
        miss = FIELDS - set(it)
        if miss:
            raise KeyError(f"{task}.json entry lacks {sorted(miss)}; fields found: {sorted(it)}. Adapt FIELDS.")
    return items


def score_pairs(img_emb, txt_emb, items):
    """Pure function (unit-tested): per-example correctness = cos(img, caption) > cos(img, negative)."""
    return [bool(img_emb[it["filename"]] @ txt_emb[it["caption"]] > img_emb[it["filename"]] @ txt_emb[it["negative_caption"]])
            for it in items]


def paired_ci(ref, test, n_boot=10000, seed=0):
    """ref, test: arrays (seeds, N) of correctness over the same examples. Returns diff, CI, seed std (pp)."""
    d = np.asarray(test, float) - np.asarray(ref, float)
    D, N = d.mean(0), d.shape[1]
    boots = D[np.random.default_rng(seed).integers(0, N, size=(n_boot, N))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sstd = float(d.mean(1).std(ddof=1)) if d.shape[0] > 1 else 0.0
    return 100 * float(D.mean()), 100 * float(lo), 100 * float(hi), 100 * sstd


def embed(model, processor, device, images, texts, coco_root, batch=128):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.model import feat
    ie, te = {}, {}
    with torch.no_grad():
        for s in range(0, len(images), batch):
            ch = images[s:s + batch]
            ims = [Image.open(os.path.join(coco_root, f)).convert("RGB") for f in ch]
            px = processor(images=ims, return_tensors="pt")["pixel_values"].to(device)
            ie.update(zip(ch, F.normalize(feat(model.get_image_features(pixel_values=px)).float(), dim=-1).cpu().numpy()))
        for s in range(0, len(texts), batch):
            ch = texts[s:s + batch]
            t = processor.tokenizer(ch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            te.update(zip(ch, F.normalize(feat(model.get_text_features(**t)).float(), dim=-1).cpu().numpy()))
    return ie, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--coco_image_root", required=True)
    ap.add_argument("--conditions", nargs="+", default=["zeroshot", "lora_baseline", "lora_hn", "lora_hn_plaus"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_dir", default=os.environ.get("PROJECT_DIR", "outputs"))
    ap.add_argument("--out", default="results/sugarcrepe.json")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N examples per task")
    args = ap.parse_args()
    import torch
    from transformers import CLIPProcessor
    from src.model import build_model
    from src.utils import load_config, load_trainable

    tasks = {t: load_task(args.data_root, t)[:args.limit] for t in TASKS}
    images = sorted({it["filename"] for its in tasks.values() for it in its})
    texts = sorted({x for its in tasks.values() for it in its for x in (it["caption"], it["negative_caption"])})
    print(f"{sum(map(len, tasks.values()))} examples, {len(images)} images, {len(texts)} texts")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, results = None, collections.defaultdict(list)
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
            ie, te = embed(model, processor, device, images, texts, args.coco_image_root)
            rec = {"seed": seed}
            for t, its in tasks.items():
                c = score_pairs(ie, te, its)
                rec[t] = {"n": len(c), "acc": float(np.mean(c)), "correct": c}
            results[cond].append(rec)
            del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"))
    print(f"\n=== SugarCrepe accuracy (%) (mean ± std over seeds; zero-shot single run) ===")
    print(f"{'condition':16s} " + " ".join(f"{t:>13s}" for t in TASKS))
    for cond in args.conditions:
        rs = results[cond]
        if rs:
            cells = [(f"{100 * np.mean([r[t]['acc'] for r in rs]):.1f}" + (f"±{100 * np.std([r[t]['acc'] for r in rs], ddof=1):.1f}" if len(rs) > 1 else "")) for t in TASKS]
            print(f"{cond:16s} " + " ".join(f"{c:>13s}" for c in cells))
    print("\n=== paired bootstrap, diff = test − ref (pp), 95% CI; seed std of diff ===")
    for ref, test in PAIRS:
        if results[ref] and results[test]:
            for t in TASKS:
                A = np.array([r[t]["correct"] for r in results[ref]]); B = np.array([r[t]["correct"] for r in results[test]])
                k = min(len(A), len(B)); d, lo, hi, sd = paired_ci(A[:k], B[:k])
                flag = "" if lo <= 0 <= hi else "  <- CI excludes 0"
                print(f"{ref:14s} -> {test:14s} {t:12s} {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]  seed std {sd:.2f}{flag}")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
