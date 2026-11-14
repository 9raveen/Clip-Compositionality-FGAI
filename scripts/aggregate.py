"""Aggregate result JSONs: mean +- std over seeds, paired bootstrap on Winoground, optional per-tag breakdown.

  python -m scripts.aggregate                        # reads $PROJECT_DIR/results, writes results/summary*.{md,csv}
  python -m scripts.aggregate --tags                 # + Winoground per-tag breakdown (downloads tags, needs HF_TOKEN)

Statistics (fixed in advance):
  * mean +- std over seeds uses the sample std (ddof=1); the zero-shot model is a single run.
  * Paired bootstrap for Winoground text/image/group between two conditions: per seed take the per-example
    correctness difference (test - ref), average over the common seeds, resample the 400 examples 10,000 times.
    The CI therefore reflects example sampling only; seed spread is reported separately.
  * A difference is called real only if the 95% CI excludes 0 AND |mean diff| exceeds the std over seeds of the
    per-seed mean difference AND, when the reference has several seeds, every test seed lies beyond the whole
    reference seed range (stops single-seed ablations from being flagged by ordinary seed noise).
    Otherwise: "no detectable difference".
  * Retrieval has no per-example data, so differences there are descriptive (no test).
"""
import argparse, collections, csv, glob, json, os
import numpy as np

W = ("text", "image", "group")
RET = ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10")
PAIRS = [("lora_baseline", "lora_hn"), ("lora_baseline", "lora_hn_plaus"), ("lora_hn_plaus", "lora_hn")]


def load(results_dir):
    runs, zs = collections.defaultdict(dict), None
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = json.load(open(p))
        if os.path.basename(p) == "zeroshot.json":
            zs = r
        elif "seed" in r and "winoground" in r:
            runs[r["config"]][r["seed"]] = r
    return runs, zs


def ms(v, scale=100.0, dec=1):
    v = np.asarray(v, float) * scale
    return f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f}" if len(v) > 1 else f"{v.mean():.{dec}f}"


def paired_bootstrap(ref, test, metric, n_boot=10000, seed=0, mask=None):
    seeds = sorted(set(ref) & set(test))
    if not seeds:
        return None
    d = np.array([np.asarray(test[s]["winoground"]["per_example"][metric], float)
                  - np.asarray(ref[s]["winoground"]["per_example"][metric], float) for s in seeds])
    if mask is not None:                       # restrict to a subset of examples (e.g. one Winoground tag)
        d = d[:, np.asarray(mask)]
    D, N = d.mean(0), d.shape[1]
    idx = np.random.default_rng(seed).integers(0, N, size=(n_boot, N))
    boots = D[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    diff = float(D.mean())
    seed_std = float(d.mean(1).std(ddof=1)) if len(seeds) > 1 else 0.0
    sel = (lambda r: float(np.mean(np.asarray(r["winoground"]["per_example"][metric], float)[np.asarray(mask)]))) if mask is not None \
        else (lambda r: r["winoground"][metric])                      # range check must use the SAME subset as the test
    ref_vals = [sel(ref[s_]) for s_ in ref]
    test_vals = [sel(test[s_]) for s_ in test]
    multi = len(ref_vals) > 1                      # with several reference seeds, test must also clear their range
    if lo > 0 and diff > seed_std and (not multi or min(test_vals) > max(ref_vals)):
        verdict = "test higher"
    elif hi < 0 and -diff > seed_std and (not multi or max(test_vals) < min(ref_vals)):
        verdict = "test lower"
    else:
        verdict = "no detectable difference"
    return {"n_seeds": len(seeds), "diff_pp": 100 * diff, "ci_lo_pp": 100 * float(lo), "ci_hi_pp": 100 * float(hi),
            "seed_std_pp": 100 * seed_std, "verdict": verdict}


def tag_breakdown(runs, tags):
    """Per-tag Winoground accuracy: mean +- std over seeds. tags: list of length N in dataset order."""
    tags = np.asarray(tags)
    out = {}
    for cfg, by_seed in runs.items():
        for t in sorted(set(tags)):
            m = tags == t
            for k in W:
                vals = [np.asarray(r["winoground"]["per_example"][k], float)[m].mean() for r in by_seed.values()]
                out[(cfg, t, k)] = (vals, int(m.sum()))
    return out


def load_tags():
    from src.eval_winoground import load_winoground
    ds = load_winoground()
    col = "collapsed_tag" if "collapsed_tag" in ds.column_names else "tag"
    return [str(x) for x in ds[col]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(os.environ.get("PROJECT_DIR", "."), "results"))
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--extra_pairs", nargs="*", default=[], help="ref:test config names, e.g. lora_baseline:lora_hn_attr")
    ap.add_argument("--tags", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs, zs = load(args.results_dir)
    pairs = PAIRS + [tuple(p.split(":")) for p in args.extra_pairs]
    md, per_seed = [], []

    md += ["## Winoground and retrieval (mean ± std over seeds, % ; zero-shot is a single run)", "",
           "| config | seeds | W text | W image | W group | i2t R@1 | R@5 | R@10 | t2i R@1 | R@5 | R@10 | val L_CLIP |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    if zs:
        rows.append(("zeroshot", [zs]))
    rows += [(c, [by[s] for s in sorted(by)]) for c, by in sorted(runs.items())]
    csv_rows = []
    for c, rs in rows:
        w = lambda k: [r["winoground"][k] for r in rs]
        t = lambda k: [r["retrieval"][k] for r in rs if "retrieval" in r]
        vc = [r["final_val"]["clip"] for r in rs if "final_val" in r]
        md.append(f"| {c} | {len(rs)} | {ms(w('text'))} | {ms(w('image'))} | {ms(w('group'))} | "
                  + " | ".join(ms(t(k)) if t(k) else "-" for k in RET) + f" | {ms(vc, 1, 3) if vc else '-'} |")
        csv_rows.append({"config": c, "n_seeds": len(rs),
                         **{f"wino_{k}_mean": float(np.mean(w(k))) for k in W},
                         **{f"wino_{k}_std": float(np.std(w(k), ddof=1)) if len(rs) > 1 else "" for k in W},
                         **{f"{k}_mean": float(np.mean(t(k))) for k in RET if t(k)}})
        for r in rs:
            for k in W:
                per_seed.append({"config": c, "seed": r.get("seed", ""), "metric": f"wino_{k}", "value": r["winoground"][k]})
            for k in RET:
                if "retrieval" in r:
                    per_seed.append({"config": c, "seed": r.get("seed", ""), "metric": k, "value": r["retrieval"][k]})
    md += ["", "val L_CLIP is comparable across conditions. Validation hn_acc is NOT in this table: it is measured on",
           "different negative files for different conditions and can be inflated by text-only shortcuts.", ""]

    md += ["## Paired bootstrap on Winoground (difference = test − ref, percentage points)", "",
           "| ref | test | metric | seeds | diff | 95% CI | seed std of diff | verdict |", "|---|---|---|---|---|---|---|---|"]
    boot_rows = []
    for ref, test in pairs:
        if ref in runs and test in runs:
            for k in W:
                b = paired_bootstrap(runs[ref], runs[test], k, args.n_boot)
                if b:
                    md.append(f"| {ref} | {test} | {k} | {b['n_seeds']} | {b['diff_pp']:+.2f} | "
                              f"[{b['ci_lo_pp']:+.2f}, {b['ci_hi_pp']:+.2f}] | {b['seed_std_pp']:.2f} | {b['verdict']} |")
                    boot_rows.append({"ref": ref, "test": test, "metric": k, **b})
    md += ["", "## Retrieval differences (descriptive only: no per-example data, no test)", "",
           "| ref | test | metric | mean diff (pp) | per-seed diff std (pp) |", "|---|---|---|---|---|"]
    for ref, test in pairs:
        if ref in runs and test in runs:
            seeds = sorted(set(runs[ref]) & set(runs[test]))
            for k in ("i2t_r1", "t2i_r1"):
                d = np.array([runs[test][s]["retrieval"][k] - runs[ref][s]["retrieval"][k] for s in seeds]) * 100
                md.append(f"| {ref} | {test} | {k} | {d.mean():+.2f} | {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f} |")

    if args.tags:
        tag_list = load_tags()
        tb = tag_breakdown(runs, tag_list)
        tags = sorted({t for (_, t, _) in tb})
        md += ["", "## Winoground by tag (mean ± std over seeds, %)", "",
               "| config | " + " | ".join(f"{t} {k}" for t in tags for k in W) + " |", "|---|" + "---|" * (len(tags) * 3)]
        for c in sorted(runs):
            md.append(f"| {c} | " + " | ".join(ms(tb[(c, t, k)][0]) for t in tags for k in W) + " |")
        md.append("| n examples | " + " | ".join(str(tb[(sorted(runs)[0], t, k)][1]) for t in tags for k in W) + " |")
        md += ["", "## Paired bootstrap by tag (diff = test − ref, pp). Tags with n < 30 are listed but must not be interpreted.", "",
               "| ref | test | tag (n) | metric | seeds | diff | 95% CI | seed std of diff | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        tg = np.asarray(tag_list)
        for ref, test in pairs:
            if ref in runs and test in runs:
                for t in tags:
                    for k in W:
                        b = paired_bootstrap(runs[ref], runs[test], k, args.n_boot, mask=(tg == t))
                        if b:
                            md.append(f"| {ref} | {test} | {t} ({int((tg == t).sum())}) | {k} | {b['n_seeds']} | {b['diff_pp']:+.2f} | "
                                      f"[{b['ci_lo_pp']:+.2f}, {b['ci_hi_pp']:+.2f}] | {b['seed_std_pp']:.2f} | {b['verdict']} |")

    open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(md) + "\n")
    for name, rws in (("summary.csv", csv_rows), ("summary_per_seed.csv", per_seed), ("bootstrap.csv", boot_rows)):
        if rws:
            keys = sorted({k for r in rws for k in r}, key=lambda k: list(rws[0]).index(k) if k in rws[0] else 99)
            with open(os.path.join(args.out_dir, name), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rws)
    print("\n".join(md))
    print(f"\nwrote summary.md, summary.csv, summary_per_seed.csv, bootstrap.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
