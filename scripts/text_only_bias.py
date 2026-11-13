"""Are the negatives detectable WITHOUT the image? Two checks per negative type.

  1. lexical : words much more frequent in negatives than in their positives (e.g. "under"). Pure Python.
  2. lm      : GPT-2 average log-likelihood. text_only_acc = fraction of pairs where the negative is LESS
               likely than its positive. ~50% means language statistics cannot separate them; ~80%+ means a
               model can often tell negatives apart from text alone (a shortcut, like the ARO critique).

  python -m scripts.text_only_bias --negatives data/negatives_train.jsonl            # lexical only
  python -m scripts.text_only_bias --negatives data/negatives_train.jsonl --lm       # + GPT-2 (needs HF download)
"""
import argparse, json, random, re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--negatives", required=True)
ap.add_argument("--lm", action="store_true")
ap.add_argument("--per_type", type=int, default=500, help="pairs sampled per type for the LM check")
ap.add_argument("--min_count", type=int, default=30)
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

pairs = defaultdict(list)
for line in open(args.negatives):
    r = json.loads(line)
    for n in r["negatives"]:
        pairs[n["type"]].append((r["caption"], n["text"]))
tok = lambda s: re.findall(r"[a-z]+", s.lower())

print("=== 1. lexical artifacts: words over-represented in negatives vs their own positives ===")
for t, ps in sorted(pairs.items()):
    pos, neg = Counter(), Counter()
    for c, n in ps:
        pos.update(set(tok(c))); neg.update(set(tok(n)))
    rows = [(w, neg[w], pos[w], (neg[w] + 1) / (pos[w] + 1)) for w in neg if neg[w] >= args.min_count]
    rows.sort(key=lambda x: -x[3])
    shown = ", ".join(f"{w} ({a} vs {b})" for w, a, b, _ in rows[:args.top] if _ > 1.3)
    print(f"{t:13s} n={len(ps):6d}  {shown or 'none above 1.3x'}")
print("(count in negatives vs in positives; swaps keep the same words so they should show none)")

if args.lm:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gtok = AutoTokenizer.from_pretrained("gpt2"); gtok.pad_token = gtok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()

    @torch.no_grad()
    def score(texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = gtok(texts[i:i + batch], return_tensors="pt", padding=True).to(dev)
            bos = torch.full((enc.input_ids.size(0), 1), gtok.bos_token_id, device=dev)
            ids = torch.cat([bos, enc.input_ids], 1)
            mask = torch.cat([torch.ones_like(bos), enc.attention_mask], 1)
            lp = torch.log_softmax(lm(input_ids=ids, attention_mask=mask).logits[:, :-1].float(), -1)
            lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1) * enc.attention_mask
            out += (lp.sum(1) / enc.attention_mask.sum(1)).tolist()
        return out

    rng = random.Random(args.seed)
    print("\n=== 2. GPT-2 text-only separability (negative less likely than its positive) ===")
    for t, ps in sorted(pairs.items()):
        sub = rng.sample(ps, min(args.per_type, len(ps)))
        sp, sn = score([c for c, _ in sub]), score([n for _, n in sub])
        acc = sum(b < a for a, b in zip(sp, sn)) / len(sub)
        print(f"{t:13s} text_only_acc = {acc:.1%}  (n={len(sub)}, 50% = not separable)")
