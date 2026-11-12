"""Build the COCO subset with leak-free image-level splits. Run ONCE on Colab (needs internet).

All images come from COCO train2017. COCO val2017 is deliberately NOT used: SugarCrepe is built on it,
so keeping it untouched keeps that benchmark unseen. Splits are by IMAGE ID, so no image (and none of its
captions) appears in two splits.

Writes (default --out data):
  captions_train.jsonl  train: --train_caps_per_image captions per image
  captions_val.jsonl    validation (loss curves, lambda choice): 2 captions per image
  captions_test.jsonl   held-out COCO retrieval test: all 5 captions per image
  splits.json           the image-id lists + seed (commit this: it is the reproducibility record)
  coco_subset/images/{id:012d}.jpg   shorter side resized to 224 (CLIP then center-crops)
  coco_subset.zip       archive of the images folder for Drive

  python scripts/prepare_coco_subset.py --num_images 8000        # default: 6,500 train / 500 val / 1,000 test
  python scripts/prepare_coco_subset.py --num_images 25000       # optional expansion
"""
import argparse, io, json, os, random, shutil, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download(url, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst + ".tmp", "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="annotations") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); bar.update(len(chunk))
    os.replace(dst + ".tmp", dst)


def fetch_image(image_id, url, out_dir, size=224, retries=3):
    path = os.path.join(out_dir, f"{image_id:012d}.jpg")
    if os.path.exists(path):
        return image_id, True
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            w, h = im.size
            s = size / min(w, h)
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
            im.save(path + ".tmp.jpg", quality=92)
            os.replace(path + ".tmp.jpg", path)
            return image_id, True
        except Exception:
            continue
    return image_id, False


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--num_images", type=int, default=8000)
    ap.add_argument("--val_images", type=int, default=500)
    ap.add_argument("--test_images", type=int, default=1000)
    ap.add_argument("--train_caps_per_image", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ann_zip = os.path.join(args.out, "raw", "annotations_trainval2017.zip")
    download(ANN_URL, ann_zip)
    d = json.load(zipfile.ZipFile(ann_zip).open("annotations/captions_train2017.json"))
    urls = {im["id"]: im["coco_url"] for im in d["images"]}
    caps = {}
    for a in d["annotations"]:
        caps.setdefault(a["image_id"], []).append(" ".join(a["caption"].split()))

    ids = sorted(caps)
    rng.shuffle(ids)
    ids = ids[:args.num_images]
    test_ids = ids[:args.test_images]
    val_ids = ids[args.test_images:args.test_images + args.val_images]
    train_ids = ids[args.test_images + args.val_images:]

    img_dir = os.path.join(args.out, "coco_subset", "images")
    os.makedirs(img_dir, exist_ok=True)
    failed = set()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch_image, i, urls[i], img_dir) for i in ids]
        for f in tqdm(as_completed(futs), total=len(futs), desc="images"):
            i, ok = f.result()
            if not ok:
                failed.add(i)
    if failed:
        print(f"[warn] {len(failed)} images failed to download and were dropped from all splits")
    train_ids, val_ids, test_ids = [[i for i in s if i not in failed] for s in (train_ids, val_ids, test_ids)]

    # leakage guard: splits must be disjoint at the image level
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids)) \
        and not (set(val_ids) & set(test_ids)), "image leakage between splits"

    def rows(split_ids, k):
        return [{"image_id": i, "caption": c} for i in split_ids for c in rng.sample(caps[i], min(k, len(caps[i])))]

    write_jsonl(os.path.join(args.out, "captions_train.jsonl"), rows(train_ids, args.train_caps_per_image))
    write_jsonl(os.path.join(args.out, "captions_val.jsonl"), rows(val_ids, 2))
    write_jsonl(os.path.join(args.out, "captions_test.jsonl"), rows(test_ids, 5))
    json.dump({"seed": args.seed, "train": train_ids, "val": val_ids, "test": test_ids},
              open(os.path.join(args.out, "splits.json"), "w"))
    print(f"train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)} images (disjoint, from COCO train2017)")
    shutil.make_archive(os.path.join(args.out, "coco_subset"), "zip", root_dir=args.out, base_dir="coco_subset")
    print("zipped", os.path.join(args.out, "coco_subset.zip"))


if __name__ == "__main__":
    main()
