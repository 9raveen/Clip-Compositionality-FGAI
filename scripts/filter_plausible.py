"""Likelihood-matched negatives: keep a negative only if it is about as plausible as its positive.

For each (caption, negative) pair keep it iff |mean-logp(neg) - mean-logp(caption)| <= tau  (GPT-2, nats/token).
Pairs that differ a lot in likelihood can be told apart from text alone (implausible swaps, rare replacement
words), which lets a model lower its hard-negative loss without looking at the image. Captions are never
dropped: a caption left with no negative simply contributes no hard-negative term (same captions as baseline B).

  python -m scripts.filter_plausible --negatives data/negatives_train.jsonl --out data/negatives_train_plaus.jsonl
  python -m scripts.filter_plausible --negatives data/negatives_val.jsonl   --out data/negatives_val_plaus.jsonl
tau is chosen from TEXT statistics only (never from benchmark results); report the value you used.
"""
import argparse, json
from collections import defaultdict


def filter_rows(rows, lp, tau):
    """Pure function (unit-tested): returns filtered rows and per-type stats."""
    kept, total, signed = defaultdict(int), defaultdict(int), defaultdict(list)
    out = []
    for r in rows:
        keep = []
        for n in r["negatives"]:
            d = lp[n["text"]] - lp[r["caption"]]
            total[n["type"]] += 1
            if abs(d) <= tau:
                keep.append(n); kept[n["type"]] += 1; signed[n["type"]].append(d)
        out.append({**r, "negatives": keep})
    return out, kept, total, signed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(args.negatives)]
    texts = sorted({r["caption"] for r in rows} | {n["text"] for r in rows for n in r["negatives"]}, key=len)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
    lp = {}
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
            bos = torch.full((enc.input_ids.size(0), 1), tok.bos_token_id, device=dev)
            ids = torch.cat([bos, enc.input_ids], 1)
            mask = torch.cat([torch.ones_like(bos), enc.attention_mask], 1)
            logp = torch.log_softmax(lm(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            logp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1) * enc.attention_mask
            for t, v in zip(chunk, (logp.sum(1) / enc.attention_mask.sum(1)).tolist()):
                lp[t] = v
    out, kept, total, signed = filter_rows(rows, lp, args.tau)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_with = sum(bool(r["negatives"]) for r in out)
    summary = {"tau": args.tau, "captions": len(out), "with_any_negative": n_with,
               "frac_with_any_negative": round(n_with / len(out), 4),
               "per_type": {t: {"kept": kept[t], "of": total[t], "retention": round(kept[t] / max(total[t], 1), 3)}
                            for t in sorted(total)}}
    json.dump(summary, open(args.out.replace(".jsonl", "_stats.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
