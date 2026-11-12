"""CLIP fine-tuning: L_total = L_CLIP + lambda * L_HN, with LoRA or full fine-tuning. Resumable.

  lambda = 0  -> standard fine-tuning baseline (positives only; same data, same adapters)
  lambda > 0  -> hard-negative fine-tuning
Usage:  python -m src.train --config configs/lora_hn.yaml --seed 0 --out_dir $PROJECT_DIR
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from src.eval_winoground import load_winoground, winoground_scores
from src.evaluate import run_eval
from src.model import build_model, feat, trainable_state
from src.utils import git_hash, load_config, load_trainable, save_ckpt, set_seed


class PairDataset(Dataset):
    """One item per IMAGE. Each epoch every image contributes one caption (and one active negative, if any).

    Caption and negative choice are deterministic functions of (index, epoch, seed), so the baseline and the
    hard-negative run see exactly the same images and captions. One caption per image per batch also avoids
    false in-batch negatives (two captions of the same image treated as a mismatch).
    """
    def __init__(self, jsonl, image_dir, neg_types, seed):
        by_img = {}
        for line in open(jsonl):
            r = json.loads(line)
            negs = [n for n in r.get("negatives", []) if neg_types is None or n["type"] in neg_types]
            by_img.setdefault(r["image_id"], []).append((r["caption"], negs))
        self.ids, self.by_img = sorted(by_img), by_img
        self.image_dir, self.seed, self.epoch = image_dir, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        caps = self.by_img[self.ids[i]]
        cap, negs = caps[(i * 7919 + self.epoch * 104729 + self.seed * 13) % len(caps)]
        neg = negs[(i * 31 + self.epoch * 17 + self.seed) % len(negs)]["text"] if negs else None
        img = Image.open(os.path.join(self.image_dir, f"{self.ids[i]:012d}.jpg")).convert("RGB")
        return img, cap, neg


def make_collate(processor):
    def collate(batch):
        imgs, caps, negs = zip(*batch)
        enc = processor(text=list(caps), images=list(imgs), return_tensors="pt", padding=True, truncation=True, max_length=77)
        mask = torch.tensor([n is not None for n in negs], dtype=torch.float)
        nt = processor.tokenizer([n if n is not None else c for n, c in zip(negs, caps)],
                                 return_tensors="pt", padding=True, truncation=True, max_length=77)
        return {"pixel_values": enc["pixel_values"], "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "neg_input_ids": nt["input_ids"], "neg_attention_mask": nt["attention_mask"], "neg_mask": mask}
    return collate


def compute_losses(img, txt, neg, neg_mask, scale, lam, hn_loss="ce", margin=0.1):
    """img/txt/neg: L2-normalised [B, D]. Returns (L_clip, L_hn, hn_acc).

    L_clip: symmetric InfoNCE over in-batch positives.
    L_hn  : for each caption that has a validated negative, rank the positive above its negative for the same image
            'ce'     -> -log sigmoid(scale * (s_pos - s_neg))   (2-way cross-entropy, no margin to tune)
            'margin' -> max(0, margin - (s_pos - s_neg)) on cosine similarities
    hn_acc: fraction of masked samples with s_pos > s_neg (a direct measure of the targeted failure).
    """
    labels = torch.arange(img.size(0), device=img.device)
    logits = scale * img @ txt.T
    l_clip = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    if neg is None:
        return l_clip, torch.zeros((), device=img.device), torch.zeros((), device=img.device)
    s_pos, s_neg = (img * txt).sum(-1), (img * neg).sum(-1)
    per = F.softplus(scale * (s_neg - s_pos)) if hn_loss == "ce" else F.relu(margin - (s_pos - s_neg))
    denom = neg_mask.sum().clamp(min=1)
    l_hn = (per * neg_mask).sum() / denom
    acc = (((s_pos > s_neg).float()) * neg_mask).sum() / denom
    return l_clip, l_hn, acc


def forward_embeddings(model, batch, need_neg):
    img = F.normalize(feat(model.get_image_features(pixel_values=batch["pixel_values"])).float(), dim=-1)
    txt = F.normalize(feat(model.get_text_features(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])).float(), dim=-1)
    neg = None
    if need_neg:
        neg = F.normalize(feat(model.get_text_features(input_ids=batch["neg_input_ids"],
                                                      attention_mask=batch["neg_attention_mask"])).float(), dim=-1)
    return img, txt, neg


@torch.no_grad()
def validate(model, dl, cfg, device, scale):
    model.eval()
    tot = {"clip": 0.0, "hn": 0.0, "hn_acc": 0.0}; n = 0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["fp16"] and device == "cuda"):
            img, txt, neg = forward_embeddings(model, batch, True)
        c, h, a = compute_losses(img, txt, neg, batch["neg_mask"], scale, 1.0, cfg["hn_loss"], cfg["margin"])
        tot["clip"] += c.item(); tot["hn"] += h.item(); tot["hn_acc"] += a.item(); n += 1
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.environ.get("PROJECT_DIR", "outputs"))
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test: stop after N steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = f"{cfg['name']}_seed{args.seed}"
    ckpt_path = os.path.join(args.out_dir, "ckpt", f"{run}_last.pt")
    result_path = os.path.join(args.out_dir, "results", f"{run}.json")
    log_path = os.path.join(args.out_dir, "logs", f"{run}.jsonl")
    for p in (ckpt_path, result_path, log_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(result_path) and args.max_steps is None:
        print(f"[skip] {result_path} exists"); return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device)
    processor = CLIPProcessor.from_pretrained(cfg["model"])
    scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else model.base_model.model.logit_scale.exp().item()
    lam = float(cfg["lambda_hn"]); need_neg = lam > 0

    neg_types = cfg.get("neg_types")
    train_ds = PairDataset(cfg["train_jsonl"], cfg["image_dir"], neg_types, args.seed)
    val_ds = PairDataset(cfg["val_jsonl"], cfg["image_dir"], neg_types, 0)
    g = torch.Generator(); g.manual_seed(args.seed)
    collate = make_collate(processor)
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=g,
                          num_workers=cfg["num_workers"], collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, cfg["warmup_steps"], total_steps)
    use_amp = cfg["fp16"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        load_trainable(model, ck["trainable"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); start_epoch = ck["epoch"] + 1
        print(f"[resume] from epoch {start_epoch}")

    wino = load_winoground()
    t0, step = time.time(), 0
    for epoch in range(start_epoch, cfg["epochs"]):
        train_ds.set_epoch(epoch); model.train()
        agg = {"clip": 0.0, "hn": 0.0, "hn_acc": 0.0}; nb = 0
        for batch in tqdm(train_dl, desc=f"epoch {epoch}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                img, txt, neg = forward_embeddings(model, batch, need_neg)
            l_clip, l_hn, acc = compute_losses(img, txt, neg, batch["neg_mask"], scale, lam, cfg["hn_loss"], cfg["margin"])
            loss = l_clip + lam * l_hn
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            agg["clip"] += l_clip.item(); agg["hn"] += l_hn.item(); agg["hn_acc"] += acc.item(); nb += 1; step += 1
            if args.max_steps and step >= args.max_steps:
                print(f"[smoke] {step} steps, {step * cfg['batch_size'] / (time.time() - t0):.1f} img/s, loss {loss.item():.4f}")
                return
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "lambda": lam, "seed": args.seed,
               "train": {k: v / nb for k, v in agg.items()}, "val": validate(model, val_dl, cfg, device, scale),
               "elapsed_min": (time.time() - t0) / 60}
        # monitoring only: never select checkpoints, lambda or epochs on Winoground
        if cfg.get("wino_every") and (epoch + 1) % cfg["wino_every"] == 0:
            rec["winoground_monitor"] = {k: v for k, v in winoground_scores(model, processor, device, ds=wino).items()}
        print(rec)
        open(log_path, "a").write(json.dumps(rec) + "\n")
        save_ckpt(ckpt_path, {"trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                              "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch})

    res = run_eval(model, processor, device, cfg, wino)           # final epoch, same protocol as zero-shot
    out = {"config": cfg["name"], "seed": args.seed, "git": git_hash(), "cfg": cfg,
           "trainable_params": sum(p.numel() for p in params), "train_time_min": (time.time() - t0) / 60,
           "final_val": validate(model, val_dl, cfg, device, scale), **res}
    json.dump(out, open(result_path, "w"), indent=2)
    print("saved", result_path)


if __name__ == "__main__":
    main()
