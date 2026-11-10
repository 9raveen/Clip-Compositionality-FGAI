"""Zero-shot / fine-tuned CLIP evaluation on Winoground.

Each example: 2 images (i0, i1) and 2 captions (c0, c1), with c0<->i0, c1<->i1.
  text score  : for each image, the correct caption scores higher
  image score : for each caption, the correct image scores higher
  group score : both text and image conditions hold
Chance: text 25%, image 25%, group 16.67%.
"""
import argparse, json, os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def load_winoground(hf_token=None):
    token = hf_token or os.environ.get("HF_TOKEN")
    try:
        return load_dataset("facebook/winoground", split="test", token=token)
    except Exception as e:  # older/newer `datasets` versions differ on loading scripts
        print(f"[warn] default load failed ({e}); retrying with trust_remote_code=True")
        return load_dataset("facebook/winoground", split="test", token=token,
                            trust_remote_code=True)


@torch.no_grad()
def winoground_scores(model, processor, device, ds=None, return_per_example=False):
    model.eval()
    ds = ds if ds is not None else load_winoground()
    t, i, g = [], [], []
    for ex in tqdm(ds, desc="Winoground"):
        imgs = [ex["image_0"].convert("RGB"), ex["image_1"].convert("RGB")]
        caps = [ex["caption_0"], ex["caption_1"]]
        inp = processor(text=caps, images=imgs, return_tensors="pt", padding=True).to(device)
        out = model(**inp)
        s = out.logits_per_image.float().cpu().numpy()  # s[image_idx, caption_idx]
        text_ok = bool(s[0, 0] > s[0, 1] and s[1, 1] > s[1, 0])
        image_ok = bool(s[0, 0] > s[1, 0] and s[1, 1] > s[0, 1])
        t.append(text_ok); i.append(image_ok); g.append(text_ok and image_ok)
    t, i, g = np.array(t), np.array(i), np.array(g)
    res = {"text": float(t.mean()), "image": float(i.mean()), "group": float(g.mean())}
    if return_per_example:
        res["per_example"] = {"text": t.tolist(), "image": i.tolist(), "group": g.tolist()}
    return res


def bootstrap_ci(correct, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    means = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--ckpt", default=None, help="optional fine-tuned checkpoint (.pt)")
    ap.add_argument("--out", default="results/zeroshot_winoground.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    res = winoground_scores(model, processor, device, return_per_example=True)
    for k in ["text", "image", "group"]:
        lo, hi = bootstrap_ci(res["per_example"][k])
        res[f"{k}_ci95"] = [lo, hi]
        print(f"{k:5s}: {res[k]*100:.1f}%  (95% CI {lo*100:.1f}-{hi*100:.1f})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "ckpt": args.ckpt, **res}, f, indent=2)
