"""Rule-based structured hard negatives for COCO captions, with a validation stage.

Input : JSONL, one {"image_id": int, "caption": str} per line
Output: JSONL, {"image_id", "caption", "negatives": [{"text", "type"}]}  (VALIDATED negatives only)

Types (what each tests)
  attr_replace  "a red car"            -> "a green car"          single attribute
  obj_replace   "a dog on a sofa"      -> "a cat on a sofa"      single object
  rel_replace   "a dog on a sofa"      -> "a dog under a sofa"   single relation
  obj_swap      "a dog chasing a cat"  -> "a cat chasing a dog"  subject/object binding
  attr_swap     "a red car, a blue bus"-> "a blue car, a red bus" attribute-object binding
Ablation categories: attribute = {attr_replace, attr_swap}; object = {obj_replace};
                     relation  = {rel_replace, obj_swap}

Every output is a CANDIDATE that must pass validate(); rejections are counted and reported.
Validation here is rule-based. Precision is estimated by a manual audit (scripts/audit_negatives.py).

Usage
  python -m src.negatives --demo
  python -m src.negatives --captions data/captions_train.jsonl --out data/negatives_train.jsonl
"""
import argparse, json, os, random, re
from collections import Counter

import spacy

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
ADJ_ANTONYM = {"big": "small", "small": "big", "large": "small", "tall": "short", "short": "tall",
               "long": "short", "young": "old", "old": "young", "open": "closed", "closed": "open",
               "empty": "full", "full": "empty", "wet": "dry", "dry": "wet", "clean": "dirty", "dirty": "clean"}
MATERIALS = ["wooden", "metal", "plastic", "stone", "glass", "leather"]

OBJECT_GROUPS = {
    "animal": ["dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["car", "truck", "bus", "bicycle", "motorcycle", "boat", "train", "airplane"],
    "furniture": ["chair", "couch", "bed", "bench"],
    "food": ["pizza", "sandwich", "cake", "banana", "apple", "donut"],
    "person": ["man", "woman", "boy", "girl"],
    "kitchen": ["cup", "bowl", "bottle", "plate", "knife", "fork", "spoon"],
    "sport": ["skateboard", "surfboard", "snowboard", "frisbee", "kite"],
}
NOUN_TO_GROUP = {n: g for g, ns in OBJECT_GROUPS.items() for n in ns}
NOUN_TO_GROUP.update({"sofa": "furniture", "bike": "vehicle", "plane": "vehicle"})
# a replacement with one of these yields a (near-)true caption, so it is never allowed
SYNONYMS = {"couch": {"sofa"}, "sofa": {"couch"}, "bike": {"bicycle", "motorcycle"}, "bicycle": {"bike"},
            "motorcycle": {"bike"}, "plane": {"airplane"}, "airplane": {"plane"}}

REL_PATTERNS = [(r"\bon top of\b", "under"), (r"\bin front of\b", "behind"),
                (r"\bto the left of\b", "to the right of"), (r"\bto the right of\b", "to the left of"),
                (r"\bleft of\b", "right of"), (r"\bright of\b", "left of")]
REL_TOKEN = {"on": "under", "under": "on", "above": "below", "below": "above", "over": "under",
             "behind": "in front of", "inside": "outside", "outside": "inside", "atop": "under"}
# directional (asymmetric) prepositions only: swapping subject/object changes the meaning
SWAP_PREPS = {"on", "under", "above", "below", "over", "behind", "inside", "outside", "atop"}
# if any of these lies between the swapped phrases the relation may be symmetric -> swap could be TRUE
SYMMETRIC = {"beside", "near", "by", "with", "and", "between", "next", "together", "each", "other", "across", "around"}
COLOR_WORDS = set(COLORS) | {"grey", "silver", "gold", "golden", "tan", "beige"}
SIZE_WORDS = {"big", "small", "large", "little", "tall", "short", "long", "tiny", "huge"}


def attr_cat(w):
    """Attribute category used to keep attr_swap between comparable attributes (color<->color, ...)."""
    if w in COLOR_WORDS:
        return "color"
    if w in SIZE_WORDS:
        return "size"
    if w in MATERIALS:
        return "material"
    return None


PERSON_COUNTERPART = {"man": "woman", "woman": "man", "boy": "girl", "girl": "boy"}   # same age class only
ARTICLES = {"a", "an", "the"}

GENERATORS = ["obj_swap", "attr_swap", "rel_replace", "attr_replace", "obj_replace"]


def fix_articles(s):
    def fn(m):
        art, nxt = m.group(1), m.group(2)
        want = "an" if nxt.lower() in "aeiou" else "a"
        return (want.capitalize() if art[0].isupper() else want) + " " + nxt
    return re.sub(r"\b([Aa]n?) ([A-Za-z])", fn, s)


def norm(s):
    return re.sub(r"[^a-z ]", "", fix_articles(s).lower()).strip()


def words(s):
    return [w for w in norm(s).split() if w not in ARTICLES]


def chunk_of(chunks, tok):
    for ch in chunks:
        if ch.start <= tok.i < ch.end:
            return ch
    return None


def _noun_chunk(chunks, tok):
    ch = chunk_of(chunks, tok)
    return ch if ch is not None and ch.root.pos_ in ("NOUN", "PROPN") else None


def swap_spans(text, a, b):
    """Swap two non-overlapping spans; returns (new_text, text_between) or None."""
    if a.start_char > b.start_char:
        a, b = b, a
    if a.end_char > b.start_char:
        return None
    ta, tb = a.text, b.text
    if a.start_char == 0 and ta[:1].isupper():
        ta, tb = ta[0].lower() + ta[1:], tb[0].upper() + tb[1:]
    between = text[a.end_char:b.start_char]
    return text[:a.start_char] + tb + between + ta + text[b.end_char:], between


def replace_span(text, start, end, new):
    return text[:start] + new + text[end:], ""


# ---------- candidate generators: each returns a list of (new_text, between_text) ----------
def agreement_ok(doc, ca, cb):
    """A swap between a singular and a plural phrase breaks verb agreement ("Carrots is slicing ..."),
    which would be a grammaticality shortcut, so reject it when the sentence has a finite verb."""
    if (ca.root.tag_ in ("NNS", "NNPS")) == (cb.root.tag_ in ("NNS", "NNPS")):
        return True
    return not any(t.tag_ in ("VBZ", "VBP") or t.lower_ in {"is", "are", "was", "were"} for t in doc)


def obj_swap(doc, text, chunks, rejected=None):
    out = []
    for tok in doc:
        pair = None
        if tok.dep_ == "prep" and tok.lower_ in SWAP_PREPS:
            obj = next((c for c in tok.children if c.dep_ == "pobj"), None)
            head, subj = tok.head, None
            if head.pos_ in ("NOUN", "PROPN"):
                subj = head
            elif head.pos_ in ("VERB", "AUX"):
                subj = next((c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                if subj is None and head.dep_ in ("acl", "relcl") and head.head.pos_ in ("NOUN", "PROPN"):
                    subj = head.head
            if obj is not None and subj is not None:
                pair = (_noun_chunk(chunks, subj), _noun_chunk(chunks, obj))
        elif tok.pos_ == "VERB":
            obj = next((c for c in tok.children if c.dep_ == "dobj"), None)
            subj = next((c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subj is None and tok.dep_ in ("acl", "relcl") and tok.head.pos_ in ("NOUN", "PROPN"):
                subj = tok.head
            if obj is not None and subj is not None:
                pair = (_noun_chunk(chunks, subj), _noun_chunk(chunks, obj))
        if pair and pair[0] is not None and pair[1] is not None \
                and pair[0].root.lower_ != pair[1].root.lower_:       # same head noun -> swap is a no-op
            if not agreement_ok(doc, *pair):
                if rejected is not None:
                    rejected[("obj_swap", "agreement_mismatch")] += 1
                continue
            res = swap_spans(text, *pair)
            if res:
                out.append(res)
    return out


def attr_swap(doc, text, chunks):
    items = []
    for ch in chunks:
        adj = next((t for t in ch if t.dep_ == "amod"), None)
        if adj is not None:
            items.append((adj, ch))
    out = []
    for i in range(len(items) - 1):
        (a, ca), (b, cb) = items[i], items[i + 1]
        if attr_cat(a.lower_) is None or attr_cat(a.lower_) != attr_cat(b.lower_):
            continue                                                  # only swap comparable attributes
        if a.lower_ != b.lower_ and ca.root.lower_ != cb.root.lower_:  # same noun -> swap would be TRUE
            res = swap_spans(text, doc[a.i:a.i + 1], doc[b.i:b.i + 1])
            if res:
                out.append(res)
    return out


def rel_replace(doc, text, rng):
    out = []
    for pat, rep in REL_PATTERNS:
        m = re.search(pat, text, flags=re.I)
        if m:
            out.append(replace_span(text, m.start(), m.end(), rep))
    for tok in doc:
        if tok.dep_ == "prep" and tok.lower_ in REL_TOKEN:
            out.append(replace_span(text, tok.idx, tok.idx + len(tok.text), REL_TOKEN[tok.lower_]))
    return out


def attr_replace(doc, text, rng):
    out = []
    for tok in doc:
        if tok.dep_ != "amod":
            continue
        w = tok.lower_
        if w in COLORS:
            new = rng.choice([c for c in COLORS if c != w])
        elif w in ADJ_ANTONYM:
            new = ADJ_ANTONYM[w]
        elif w in MATERIALS:
            new = rng.choice([m for m in MATERIALS if m != w])
        else:
            continue
        out.append(replace_span(text, tok.idx, tok.idx + len(tok.text), new))
    return out


def obj_replace(doc, text, rng):
    out = []
    for tok in doc:
        if tok.tag_ != "NN":
            continue
        g = NOUN_TO_GROUP.get(tok.lower_)
        if g is None:
            continue
        banned = SYNONYMS.get(tok.lower_, set()) | {tok.lower_}
        pool = [n for n in OBJECT_GROUPS[g] if n not in banned]
        if g == "person":
            pool = [PERSON_COUNTERPART[tok.lower_]]
        out.append(replace_span(text, tok.idx, tok.idx + len(tok.text), rng.choice(pool)))
    return out


# ---------- validation stage ----------
def validate(caption, neg, typ, between=""):
    """Return None if the candidate passes, else a rejection reason."""
    if norm(neg) == norm(caption):
        return "identical"
    if typ in ("obj_swap", "attr_swap"):
        if sorted(words(caption)) != sorted(words(neg)):
            return "not_a_permutation"
        if "each other" in caption.lower():
            return "reciprocal"
    if typ == "obj_swap" and any(w in SYMMETRIC for w in re.findall(r"[a-z]+", between.lower())):
        return "symmetric_relation"
    if typ == "obj_swap" and len(caption.split()) > 16:
        return "too_complex"                     # parser errors are common on long captions
    if len(neg.split()) > 40:
        return "too_long"
    return None


def make_negatives(doc, text, rng, rejected=None):
    chunks = list(doc.noun_chunks)
    raw = {"obj_swap": obj_swap(doc, text, chunks, rejected), "attr_swap": attr_swap(doc, text, chunks),
           "rel_replace": rel_replace(doc, text, rng), "attr_replace": attr_replace(doc, text, rng),
           "obj_replace": obj_replace(doc, text, rng)}
    result, seen = [], {norm(text)}
    for typ in GENERATORS:
        valid = []
        for cand, between in raw[typ]:
            cand = fix_articles(cand).strip()
            reason = validate(text, cand, typ, between)
            if reason is None and norm(cand) in seen:
                reason = "duplicate"
            if reason is not None:
                if rejected is not None:
                    rejected[(typ, reason)] += 1
                continue
            valid.append(cand)
        if valid:
            pick = rng.choice(valid)
            seen.add(norm(pick))
            result.append({"text": pick, "type": typ})
    return result


DEMO = ["A dog on a sofa.", "A man riding a horse.", "A red car next to a blue bike.",
        "A woman holding an umbrella.", "A cat sitting under a wooden table.",
        "A red car and a blue car parked outside.", "A cat beside a dog.", "A couch in a living room.",
        "A white plate with a slice of pizza on it."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/captions_train.jsonl")
    ap.add_argument("--out", default="data/negatives_train.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_process", type=int, default=2)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    rng = random.Random(args.seed)
    if args.demo:
        rej = Counter()
        for cap, doc in zip(DEMO, nlp.pipe(DEMO)):
            print(cap)
            for n in make_negatives(doc, cap, rng, rej):
                print(f"    [{n['type']:12s}] {n['text']}")
        print("rejected:", dict(rej))
        return

    rows = [json.loads(l) for l in open(args.captions)]
    if args.limit:
        rows = rows[:args.limit]
    texts = [r["caption"].strip() for r in rows]
    kept, rejected, examples, n_with = Counter(), Counter(), {t: [] for t in GENERATORS}, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r, text, doc in zip(rows, texts, nlp.pipe(texts, batch_size=256, n_process=args.n_process)):
            negs = make_negatives(doc, text, rng, rejected)
            for n in negs:
                kept[n["type"]] += 1
                if len(examples[n["type"]]) < 5:
                    examples[n["type"]].append((text, n["text"]))
            n_with += bool(negs)
            f.write(json.dumps({"image_id": r["image_id"], "caption": text, "negatives": negs}) + "\n")
    total = len(rows)
    summary = {"captions": total, "with_any_negative": n_with, "frac_with_any_negative": round(n_with / total, 4),
               "validated_per_type": {t: {"count": kept[t], "frac": round(kept[t] / total, 4)} for t in GENERATORS},
               "rejected": {f"{t}:{r}": c for (t, r), c in sorted(rejected.items())}}
    with open(args.out.replace(".jsonl", "_stats.json"), "w") as f:
        json.dump({"summary": summary, "examples": examples}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
