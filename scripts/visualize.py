"""Figures from files committed in the repo (results/*.json, logs/*.jsonl). Run from a fresh clone:

  python -m scripts.visualize                      # writes figures/*.png
Reads: results/<config>_seed<k>.json, results/zeroshot.json, results/sugarcrepe.json, results/probe_hn.json,
       logs/<config>_seed<k>.jsonl. Missing inputs skip their figure. Error bars = std over seeds (sample std);
Every dot is one seed. Nothing is hard-coded: all numbers come from the files.
"""
import argparse, collections, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN = ["zeroshot", "lora_baseline", "lora_hn", "lora_hn_plaus"]
LABEL = {"zeroshot": "A zero-shot", "lora_baseline": "B LoRA baseline", "lora_hn": "C hard negatives (all)",
         "lora_hn_plaus": "C-plaus matched negs", "lora_hn_attr": "attribute only", "lora_hn_obj": "object only",
         "lora_hn_rel": "relation only", "lora_hn_lam05": "lambda 0.5", "lora_hn_lam2": "lambda 2",
         "lora_hn_objswap": "obj_swap only", "lora_hn_relrep": "rel_replace only"}
COLOR = {"zeroshot": "#9a9a9a", "lora_baseline": "#4c78a8", "lora_hn": "#e45756", "lora_hn_plaus": "#54a24b",
         "lora_hn_attr": "#f2b134", "lora_hn_obj": "#b279a2", "lora_hn_rel": "#ff9da6", "lora_hn_lam05": "#79706e", "lora_hn_lam2": "#9d7660",
         "lora_hn_objswap": "#17becf", "lora_hn_relrep": "#bcbd22"}
ABL = ["lora_baseline", "lora_hn", "lora_hn_plaus", "lora_hn_attr", "lora_hn_obj", "lora_hn_rel", "lora_hn_objswap", "lora_hn_relrep",
       "lora_hn_lam05", "lora_hn_lam2"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def load_runs(rd):
    runs = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(rd, "*_seed*.json")):
        r = json.load(open(p))
        if "winoground" in r and "seed" in r:
            runs[r["config"]][r["seed"]] = r
    zs = os.path.join(rd, "zeroshot.json")
    if os.path.exists(zs):
        runs["zeroshot"] = {0: json.load(open(zs))}
    return runs


def bars(ax, groups, series, getter, colors, width=0.8):
    """groups: x categories; series: list of condition keys; getter(cond, group) -> list of per-seed values (in %)."""
    n = len(series); w = width / n
    for i, c in enumerate(series):
        for j, g in enumerate(groups):
            v = getter(c, g)
            if not v:
                continue
            x = j - width / 2 + w * (i + 0.5)
            ax.bar(x, np.mean(v), w * 0.92, color=colors[c], label=LABEL[c] if j == 0 else None,
                   yerr=np.std(v, ddof=1) if len(v) > 1 else None, capsize=2, ecolor="black", error_kw={"lw": 0.8})
            if len(v) > 1:
                ax.scatter([x] * len(v), v, s=6, color="black", zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)


def fig_winoground(runs, out):
    series = [c for c in MAIN if c in runs]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars(ax, ["text", "image", "group"], series, lambda c, g: [100 * r["winoground"][g] for r in runs[c].values()], COLOR)
    for j, ch in enumerate([25, 25, 16.67]):
        ax.hlines(ch, j - 0.42, j + 0.42, colors="black", linestyles="dashed", lw=0.9)
    ax.plot([], [], "k--", lw=0.9, label="chance")
    ax.set_ylabel("Winoground score (%)"); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Winoground (mean ± std over seeds; dots = seeds)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_retrieval(runs, out):
    series = [c for c in MAIN if c in runs]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, d, t in zip(axes, ("i2t", "t2i"), ("image → text", "text → image")):
        bars(ax, ["R@1", "R@5", "R@10"], series,
             lambda c, g, d=d: [100 * r["retrieval"][f"{d}_r{g[2:]}"] for r in runs[c].values() if "retrieval" in r], COLOR)
        ax.set_title(t); ax.set_ylim(40, 100)
    axes[0].set_ylabel("Recall (%) on 1,000 held-out COCO images"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_sugarcrepe(sc, out):
    series = [c for c in MAIN if c in sc]
    tasks = [t for t in sc[series[0]][0] if t != "seed"]
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    bars(ax, tasks, series, lambda c, g: [100 * r[g]["acc"] for r in sc[c]], COLOR)
    ax.set_ylim(50, 100); ax.set_ylabel("accuracy (%)"); ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("SugarCrepe (COCO val2017 images, never trained on)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_ablation(runs, sc, out):
    cfgs = [c for c in ABL if c in runs]
    panels = [("Winoground text (%)", lambda c: [100 * r["winoground"]["text"] for r in runs[c].values()]),
              ("Winoground group (%)", lambda c: [100 * r["winoground"]["group"] for r in runs[c].values()]),
              ("i2t R@1 (%)", lambda c: [100 * r["retrieval"]["i2t_r1"] for r in runs[c].values() if "retrieval" in r])]
    if sc:
        panels.insert(2, ("SugarCrepe replace_rel (%)", lambda c: [100 * r["replace_rel"]["acc"] for r in sc[c]] if c in sc else []))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), sharey=False)
    for ax, (title, get) in zip(np.atleast_1d(axes), panels):
        for i, c in enumerate(cfgs):
            v = get(c)
            if not v:
                continue
            ax.bar(i, np.mean(v), 0.75, color=COLOR[c], yerr=np.std(v, ddof=1) if len(v) > 1 else None, capsize=2)
            ax.scatter([i] * len(v), v, s=7, color="black", zorder=3)
        ax.set_xticks(range(len(cfgs))); ax.set_xticklabels([f"{LABEL[c]} (n={len(runs[c])})" for c in cfgs], rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        vals = [x for c in cfgs for x in get(c)]
        if vals:
            lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.25 + 0.5
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Ablations (n = number of seeds; dots = seeds; y-axes truncated, so compare differences, not bar heights)", fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_probe(probe, out):
    key = next(k for k in next(iter(probe.values())) if "plaus" not in k)
    types = ["attr_replace", "attr_swap", "obj_replace", "obj_swap", "rel_replace"]
    series = [c for c in MAIN if c in probe]
    def get(c, t):
        return [100 * r[t]["hn_acc"] for r in probe[c][key] if t in r]
    n = {t: next((r[t]["n"] for c in series for r in probe[c][key] if t in r), 0) for t in types}
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars(ax, types, series, get, COLOR)
    ax.set_xticklabels([f"{t}\n(n={n[t]})" for t in types]); ax.axhline(50, color="black", ls="--", lw=0.8)
    ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("caption ranked above its negative (%)"); ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center")
    ax.set_title("Hard-negative probe on validation pairs (attr_swap has too few pairs to interpret)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_curves(logs_dir, out):
    cur = collections.defaultdict(lambda: collections.defaultdict(dict))        # cond -> seed -> epoch -> rec
    for p in glob.glob(os.path.join(logs_dir, "*_seed*.jsonl")):
        m = re.match(r"(.+)_seed(\d+)\.jsonl$", os.path.basename(p))
        for line in open(p):
            r = json.loads(line); cur[m.group(1)][int(m.group(2))][r["epoch"]] = r
    series = [c for c in MAIN if c in cur]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, split in zip(axes, ("train", "val")):
        for c in series:
            ep = sorted(set.intersection(*[set(v) for v in cur[c].values()]))
            y = np.array([[cur[c][s][e][split]["clip"] for e in ep] for s in cur[c]])
            x = np.array(ep) + 1
            ax.plot(x, y.mean(0), color=COLOR[c], label=LABEL[c])
            if len(y) > 1:
                ax.fill_between(x, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), color=COLOR[c], alpha=0.2)
        ax.set_title(f"{split} L_CLIP"); ax.set_xlabel("epoch")
    axes[0].set_ylabel("contrastive loss"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--out_dir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    runs = load_runs(a.results_dir)
    sc = json.load(open(p)) if os.path.exists(p := os.path.join(a.results_dir, "sugarcrepe.json")) else None
    probe = json.load(open(p)) if os.path.exists(p := os.path.join(a.results_dir, "probe_hn.json")) else None
    jobs = [("winoground.png", lambda o: fig_winoground(runs, o), bool(runs)),
            ("retrieval.png", lambda o: fig_retrieval(runs, o), bool(runs)),
            ("sugarcrepe.png", lambda o: fig_sugarcrepe(sc, o), bool(sc)),
            ("ablation.png", lambda o: fig_ablation(runs, sc, o), bool(runs)),
            ("probe.png", lambda o: fig_probe(probe, o), bool(probe)),
            ("loss_curves.png", lambda o: fig_curves(a.logs_dir, o), os.path.isdir(a.logs_dir))]
    for name, fn, ok in jobs:
        if ok:
            fn(os.path.join(a.out_dir, name)); print("wrote", os.path.join(a.out_dir, name))
        else:
            print("skipped", name, "(input missing)")


if __name__ == "__main__":
    main()
