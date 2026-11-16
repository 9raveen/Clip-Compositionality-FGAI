"""Qualitative Winoground examples selected by a FIXED rule (no hand-picking).

A condition counts as "correct" on an example if a majority of its seeds are correct on the chosen metric.
We list, in dataset order, the first N examples where ref is right and test is wrong, and the first N where test is
right and ref is wrong. Both directions are shown and their total counts are printed, so the figure is not one-sided.
Only per-example correctness is shown (similarity scores would need a GPU re-run).

  python -m scripts.qualitative                                   # B vs C, text score
  python -m scripts.qualitative --test lora_hn_plaus --n 4
"""
import argparse, collections, glob, json, os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"lora_baseline": "B baseline", "lora_hn": "C hard negatives (all)", "lora_hn_plaus": "C-plaus matched negatives"}


def load_runs(rd, config):
    out = {}
    for p in glob.glob(os.path.join(rd, f"{config}_seed*.json")):
        r = json.load(open(p))
        if r.get("config") == config and "winoground" in r:
            out[r["seed"]] = r
    return out


def votes(runs, metric, i):
    v = [bool(r["winoground"]["per_example"][metric][i]) for r in runs.values()]
    return sum(v), len(v)


def select(ref, test, metric):
    N = len(next(iter(ref.values()))["winoground"]["per_example"][metric])
    a, b = [], []
    for i in range(N):
        (rv, rn), (tv, tn) = votes(ref, metric, i), votes(test, metric, i)
        r_ok, t_ok = rv > rn / 2, tv > tn / 2
        if r_ok and not t_ok:
            a.append(i)
        elif t_ok and not r_ok:
            b.append(i)
    return a, b


def draw(ds, idx, ref, test, ref_name, test_name, metric, title, out):
    n = len(idx)
    if n == 0:
        print("no examples for", title); return
    fig, axes = plt.subplots(n, 2, figsize=(7.2, 3.35 * n), squeeze=False)
    for row, i in enumerate(idx):
        ex = ds[i]
        tag = ex.get("collapsed_tag") or ex.get("tag") or "?"
        (rv, rn), (tv, tn) = votes(ref, metric, i), votes(test, metric, i)
        for col in (0, 1):
            ax = axes[row][col]
            ax.imshow(ex[f"image_{col}"].convert("RGB")); ax.axis("off")
            ax.set_title(textwrap.fill(f'image {col}: "{ex[f"caption_{col}"]}"', 44), fontsize=8)
        axes[row][0].text(0, -0.10, f"example #{i}, tag {tag} | {LABEL.get(ref_name, ref_name)}: {rv}/{rn} seeds correct | "
                          f"{LABEL.get(test_name, test_name)}: {tv}/{tn}", transform=axes[row][0].transAxes, fontsize=7.5, va="top")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="lora_baseline")
    ap.add_argument("--test", default="lora_hn")
    ap.add_argument("--metric", default="text", choices=["text", "image", "group"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    from src.eval_winoground import load_winoground
    ref, test = load_runs(args.results_dir, args.ref), load_runs(args.results_dir, args.test)
    assert ref and test, "result files not found"
    ds = load_winoground()
    a, b = select(ref, test, args.metric)
    print(f"{args.metric}: {len(a)} examples where {args.ref} is right and {args.test} wrong; {len(b)} the other way round")
    os.makedirs(args.out_dir, exist_ok=True)
    draw(ds, a[:args.n], ref, test, args.ref, args.test, args.metric,
         f"{LABEL.get(args.ref, args.ref)} right, {LABEL.get(args.test, args.test)} wrong ({args.metric} score; first {args.n} of {len(a)} in dataset order)",
         os.path.join(args.out_dir, f"qualitative_{args.metric}_{args.ref}_right_{args.test}_wrong.png"))
    draw(ds, b[:args.n], ref, test, args.ref, args.test, args.metric,
         f"{LABEL.get(args.test, args.test)} right, {LABEL.get(args.ref, args.ref)} wrong ({args.metric} score; first {args.n} of {len(b)} in dataset order)",
         os.path.join(args.out_dir, f"qualitative_{args.metric}_{args.test}_right_{args.ref}_wrong.png"))


if __name__ == "__main__":
    main()
