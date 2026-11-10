"""Hard-negative fine-tuning of CLIP. Resumable: re-running the same command continues.

Usage:
  python -m src.train --config configs/full_all_negs.yaml --seed 0 --out_dir $PROJECT_DIR
"""
import argparse, json, os, random, time
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup

from src.eval_winoground import load_winoground, winoground_scores
from src.utils import load_ckpt_if_exists, load_config, save_ckpt, set_seed


class CocoNegDataset(Dataset):
    """Reads negatives.jsonl:
    {"image_id": 391895, "caption": "...", "negatives": [{"text": "...", "type": "obj_swap"}]}
    """
    def __init__(self, jsonl_path, image_dir, use_negatives, neg_types=None, max_samples=None):
        self.image_dir, self.use_negatives = image_dir, use_negatives
        self.items = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                negs = [n["text"] for n in r.get("negatives", [])
                        if neg_types is None or n["type"] in neg_types]
                if use_negatives and not negs:
                    continue  # only captions that have a negative
                self.items.append((r["image_id"], r["caption"], negs))
        if max_samples:
            rng = random.Random(1234)  # fixed subset independent of the training seed
            rng.shuffle(self.items)
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        image_id, cap, negs = self.items[idx]
        img = Image.open(os.path.join(self.image_dir, f"{image_id:012d}.jpg")).convert("RGB")
        neg = random.choice(negs) if (self.use_negatives and negs) else None
        return img, cap, neg


def make_collate(processor, use_negatives):
    def collate(batch):
        imgs, caps, negs = zip(*batch)
        enc = processor(text=list(caps), images=list(imgs), return_tensors="pt",
                        padding=True, truncation=True, max_length=77)
        out = {"pixel_values": enc["pixel_values"], "input_ids": enc["input_ids"],
               "attention_mask": enc["attention_mask"]}
        if use_negatives:
            nt = processor.tokenizer(list(negs), return_tensors="pt", padding=True,
                                     truncation=True, max_length=77)
            out["neg_input_ids"], out["neg_attention_mask"] = nt["input_ids"], nt["attention_mask"]
        return out
    return collate


def clip_loss(img, txt, neg, scale):
    """Symmetric InfoNCE; hard-negative captions added as extra i2t candidates."""
    labels = torch.arange(img.size(0), device=img.device)
    cand = txt if neg is None else torch.cat([txt, neg])
    l_i2t = F.cross_entropy(scale * img @ cand.T, labels)
    l_t2i = F.cross_entropy(scale * txt @ img.T, labels)
    return (l_i2t + l_t2i) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.environ.get("PROJECT_DIR", "outputs"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = f"{cfg['name']}_seed{args.seed}"
    ckpt_path = os.path.join(args.out_dir, "ckpt", f"{run}_last.pt")
    result_path = os.path.join(args.out_dir, "results", f"{run}.json")
    log_path = os.path.join(args.out_dir, "logs", f"{run}.jsonl")
    for p in (ckpt_path, result_path, log_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(result_path):
        print(f"[skip] {result_path} already exists"); return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(cfg["model"]).to(device)
    processor = CLIPProcessor.from_pretrained(cfg["model"])

    ds = CocoNegDataset(cfg["negatives_file"], cfg["image_dir"], cfg["use_negatives"],
                        cfg.get("neg_types"), cfg.get("max_samples"))
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=g,
                    num_workers=cfg["num_workers"], collate_fn=make_collate(processor, cfg["use_negatives"]))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(dl)
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_steps"], total)
    use_amp = cfg["fp16"] and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = load_ckpt_if_exists(ckpt_path, model, opt, sched, scaler, device)
    wino = load_winoground()
    t0 = time.time()

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        running = []
        for batch in tqdm(dl, desc=f"epoch {epoch}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                img = F.normalize(model.get_image_features(pixel_values=batch["pixel_values"]), dim=-1)
                txt = F.normalize(model.get_text_features(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"]), dim=-1)
                neg = None
                if "neg_input_ids" in batch:
                    neg = F.normalize(model.get_text_features(input_ids=batch["neg_input_ids"],
                                      attention_mask=batch["neg_attention_mask"]), dim=-1)
                loss = clip_loss(img, txt, neg, model.logit_scale.exp().clamp(max=100))
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            running.append(loss.item())

        # NOTE: Winoground is logged for monitoring only. Do NOT select checkpoints on it.
        w = winoground_scores(model, processor, device, ds=wino)
        rec = {"epoch": epoch, "train_loss": sum(running) / len(running), "winoground": w}
        print(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        save_ckpt(ckpt_path, model, opt, sched, scaler, epoch)

    final = winoground_scores(model, processor, device, ds=wino)
    # TODO(C): add ARO, SugarCrepe, COCO/Flickr30k retrieval, ImageNet zero-shot here
    with open(result_path, "w") as f:
        json.dump({"config": cfg["name"], "seed": args.seed, "winoground": final,
                   "train_time_min": (time.time() - t0) / 60}, f, indent=2)
    print("saved", result_path)


if __name__ == "__main__":
    main()
