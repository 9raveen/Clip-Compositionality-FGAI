"""Image-text retrieval Recall@K on the held-out COCO test split (a sanity check on what fine-tuning cost)."""
import json, os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.model import feat


@torch.no_grad()
def retrieval_recall(model, processor, device, jsonl, image_dir, ks=(1, 5, 10), batch=128, max_images=None):
    model.eval()
    by_img = {}
    for line in open(jsonl):
        r = json.loads(line)
        by_img.setdefault(r["image_id"], []).append(r["caption"])
    ids = sorted(by_img)[:max_images]
    texts, owner = [], []
    for k, i in enumerate(ids):
        for c in by_img[i]:
            texts.append(c); owner.append(k)
    owner = torch.tensor(owner, device=device)

    img_emb = []
    for s in tqdm(range(0, len(ids), batch), desc="retrieval: images"):
        ims = [Image.open(os.path.join(image_dir, f"{i:012d}.jpg")).convert("RGB") for i in ids[s:s + batch]]
        px = processor(images=ims, return_tensors="pt")["pixel_values"].to(device)
        img_emb.append(F.normalize(feat(model.get_image_features(pixel_values=px)).float(), dim=-1))
    txt_emb = []
    for s in range(0, len(texts), batch):
        t = processor.tokenizer(texts[s:s + batch], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        txt_emb.append(F.normalize(feat(model.get_text_features(**t)).float(), dim=-1))
    img_emb, txt_emb = torch.cat(img_emb), torch.cat(txt_emb)
    sim = txt_emb @ img_emb.T                                   # [num_texts, num_images]

    gt_sim = sim[torch.arange(len(texts), device=device), owner]
    t2i_rank = (sim > gt_sim[:, None]).sum(1)                   # images ranked above the true one
    i2t_rank = []
    for k in range(len(ids)):
        col = sim[:, k]
        best_gt = col[owner == k].max()
        i2t_rank.append((col > best_gt).sum())
    i2t_rank = torch.stack(i2t_rank)
    res = {"num_images": len(ids), "num_texts": len(texts)}
    for k in ks:
        res[f"i2t_r{k}"] = float((i2t_rank < k).float().mean())
        res[f"t2i_r{k}"] = float((t2i_rank < k).float().mean())
    return res
