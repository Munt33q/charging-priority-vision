"""Prepare MIO-TCD in YOLO format.

Primary source: Kaggle mirror `mzrare21/mio-tcd-yolo` (68,661 imgs, already
YOLO, official train/val/test split, classes: car, pickup_truck,
single_unit_truck, articulated_truck, bus, motorcycle).
Fallback: official tar from tcd.miovision.com (csv annotations).

Only classes that map to tiers are kept. Filenames are tagged with the
official split (mitcd_train_..., mitcd_val_..., mitcd_test_...) so the merge
step can preserve it.

Output: data/processed/miotcd_yolo/{images,labels}
"""
import os
import sys
import csv
import glob
import json
import shutil
import collections

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, RAW_DIR, PROC_DIR  # noqa: E402

OUT_DIR = os.path.join(PROC_DIR, "miotcd_yolo")
MIRROR = os.path.join(RAW_DIR, "mio-tcd-yolo")
TAR_PATH = os.path.join(RAW_DIR, "MIO-TCD-Localization.tar")

# mirror class order (dataset.yaml)
MIRROR_MAP = {
    "car": "car",
    "pickup_truck": "pickup_truck",
    "single_unit_truck": "truck",
    "articulated_truck": "semi_truck",
    "bus": None,
    "motorcycle": None,
}
SPLIT_DIRS = {"train": "train", "val": "val", "test": "test"}


def from_mirror():
    """Fast path: images were already bulk-copied by robocopy to
    data/processed/miotcd_stage (same layout as the mirror). We rewrite the
    small label files and rename images (metadata-only, same volume)."""
    import yaml
    stage = os.path.join(PROC_DIR, "miotcd_stage")
    img_src = stage if os.path.isdir(os.path.join(stage, "images", "train")) else MIRROR
    with open(os.path.join(MIRROR, "dataset.yaml"), encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = [n.lower() for n in meta["names"]]
    id_map = {i: (CONFIG["CLASS_NAMES"].index(MIRROR_MAP[n])
                  if MIRROR_MAP.get(n) else None)
              for i, n in enumerate(names)}
    print("mirror classes:", names)
    print("id map:", id_map, "| image source:", img_src)

    counts = collections.Counter()
    n_out = 0
    for split, sub in SPLIT_DIRS.items():
        lbl_dir = os.path.join(img_src, "labels", sub)
        img_dir = os.path.join(img_src, "images", sub)
        out_img = os.path.join(OUT_DIR, "images")
        out_lbl = os.path.join(OUT_DIR, "labels")
        n_split = 0
        for lp in glob.glob(os.path.join(lbl_dir, "*.txt")):
            stem = os.path.splitext(os.path.basename(lp))[0]
            out_lines = []
            with open(lp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    new_id = id_map.get(int(parts[0]))
                    if new_id is not None:
                        out_lines.append(f"{new_id} " + " ".join(parts[1:]))
                        counts[CONFIG["CLASS_NAMES"][new_id]] += 1
            if not out_lines:
                continue
            src_img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.isfile(cand):
                    src_img = cand
                    break
            if src_img is None:
                continue
            base = f"mitcd_{split}_{stem}"
            os.rename(src_img, os.path.join(out_img, base + os.path.splitext(src_img)[1]))
            with open(os.path.join(out_lbl, base + ".txt"), "w") as f:
                f.write("\n".join(out_lines))
            n_split += 1
        print(f"  {split}: {n_split} images kept")
        n_out += n_split
    return n_out, counts


def from_official_tar():
    """Fallback: parse the official tar's csv annotations."""
    import tarfile
    extract = os.path.join(RAW_DIR, "MIO-TCD-Localization")
    if not os.path.isdir(extract) or not any(os.scandir(extract)):
        print(f"extracting {TAR_PATH} (~3.7 GB) ...")
        os.makedirs(extract, exist_ok=True)
        with tarfile.open(TAR_PATH, "r") as tf:
            for m in tf:
                tf.extract(m, extract)

    import cv2
    img_index = {}
    for root, _, files in os.walk(extract):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                img_index[os.path.splitext(f)[0].lower()] = os.path.join(root, f)
    csvs = [p for p in glob.glob(os.path.join(extract, "**", "*.csv"), recursive=True)]
    print("csvs:", csvs)
    keep = set(CONFIG["MIO_TCD_MAP"].keys())
    counts = collections.Counter()
    n_out = 0
    for csv_path in csvs:
        rows_by_img = collections.defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]
            def col(*names):
                for n in names:
                    if n in header:
                        return header.index(n)
                raise ValueError(f"{names} not in {header}")
            ci, cl = col("image_id", "image", "filename"), col("label", "class")
            cx1, cy1 = col("xmin", "x1"), col("ymin", "y1")
            cx2, cy2 = col("xmax", "x2"), col("ymax", "y2")
            for row in reader:
                if len(row) <= max(ci, cl, cx1, cy1, cx2, cy2):
                    continue
                label = row[cl].strip().lower().replace(" ", "_")
                if label not in keep:
                    continue
                rows_by_img[row[ci].strip()].append(
                    (label, *map(float, (row[cx1], row[cy1], row[cx2], row[cy2]))))
        for img_id, boxes in rows_by_img.items():
            src = img_index.get(img_id.lower())
            if src is None:
                continue
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for label, x1, y1, x2, y2 in boxes:
                x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
                y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                unified = CONFIG["MIO_TCD_MAP"][label]
                lines.append(f"{CONFIG['CLASS_NAMES'].index(unified)} "
                             f"{(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                             f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                counts[unified] += 1
            if not lines:
                continue
            base = f"mitcd_train_{os.path.splitext(os.path.basename(src))[0]}"
            cv2.imwrite(os.path.join(OUT_DIR, "images", base + ".jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(OUT_DIR, "labels", base + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_out += 1
    return n_out, counts


def main():
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    if os.path.isdir(os.path.join(MIRROR, "images", "train")):
        print("using Kaggle YOLO mirror")
        n, counts = from_mirror()
    elif os.path.isfile(TAR_PATH):
        print("using official tar")
        n, counts = from_official_tar()
    else:
        raise RuntimeError("no MIO-TCD source found (mirror or tar)")
    print(f"\nwrote {n} images -> {OUT_DIR}")
    print("boxes per class:", dict(counts))


if __name__ == "__main__":
    main()
