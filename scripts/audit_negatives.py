"""Manual audit of negative validity: the evidence that the validation stage works.

  python -m scripts.audit_negatives sample --negatives data/negatives_train.jsonl --n 40 --out audit.csv
  -> open audit.csv (Sheets/Excel), fill column `label`: 1 = clearly FALSE for the image (valid negative),
     0 = could be TRUE / ungrammatical / ambiguous. Split the rows between teammates.
  python -m scripts.audit_negatives score --csv audit.csv
  -> precision per negative type with a 95% Wilson interval; report it in the README.
"""
import argparse, csv, json, math, random
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("sample"); s.add_argument("--negatives", required=True); s.add_argument("--n", type=int, default=40)
s.add_argument("--out", default="audit.csv"); s.add_argument("--seed", type=int, default=0)
c = sub.add_parser("score"); c.add_argument("--csv", required=True)
args = ap.parse_args()

if args.cmd == "sample":
    by_type = defaultdict(list)
    for line in open(args.negatives):
        r = json.loads(line)
        for n in r["negatives"]:
            by_type[n["type"]].append((r["image_id"], r["caption"], n["text"], n["type"]))
    rng = random.Random(args.seed)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["type", "image_id", "caption", "negative", "label"])
        for t, items in sorted(by_type.items()):
            for iid, cap, neg, typ in rng.sample(items, min(args.n, len(items))):
                w.writerow([typ, iid, cap, neg, ""])
    print("wrote", args.out)
else:
    tot = defaultdict(lambda: [0, 0])
    for row in csv.DictReader(open(args.csv)):
        if row["label"].strip() in ("0", "1"):
            tot[row["type"]][1] += 1; tot[row["type"]][0] += int(row["label"])
    for t, (k, n) in sorted(tot.items()):
        p, lo, hi = wilson(k, n)
        print(f"{t:13s} valid {k}/{n} = {p:.0%}  (95% CI {lo:.0%}-{hi:.0%})")
