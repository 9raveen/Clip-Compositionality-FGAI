"""Fast interactive labelling for the negative audit. Run inside a Colab/Jupyter CELL (input() does not work with !python):

    from scripts.label_audit import label
    label(f"{PROJECT_DIR}/audit_p1.csv", per_type=20)

Judge each pair from the TWO CAPTIONS ONLY (no image): 1 = the negative is clearly FALSE whenever the caption is true;
0 = it could be true, is ambiguous, or is ungrammatical/nonsensical. s = skip, q = save and quit. Progress is saved after every label,
so you can stop and resume. Then score with:  python -m scripts.audit_negatives score --csv <file>
"""
import csv, os


def _save(path, rows, fields):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def label(path, per_type=20):
    rows = list(csv.DictReader(open(path)))
    fields = list(rows[0].keys())
    done = {}
    for r in rows:
        if r["label"].strip() in ("0", "1"):
            done[r["type"]] = done.get(r["type"], 0) + 1
    for r in rows:
        t = r["type"]
        if r["label"].strip() in ("0", "1") or done.get(t, 0) >= per_type:
            continue
        print(f"\n[{t}]  ({done.get(t, 0) + 1}/{per_type})\n  caption : {r['caption']}\n  negative: {r['negative']}")
        while True:
            a = input("  1 = clearly false | 0 = could be true/ambiguous/ungrammatical | s skip | q quit > ").strip().lower()
            if a in ("0", "1", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue
        r["label"] = a
        done[t] = done.get(t, 0) + 1
        _save(path, rows, fields)
    _save(path, rows, fields)
    print("\nlabelled per type:", done)
