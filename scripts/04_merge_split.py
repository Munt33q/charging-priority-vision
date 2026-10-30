"""Merge all prepared YOLO subsets into train/val/test using Ultralytics
txt-file lists (no image copying - paths referenced in place).

- Emergency dataset: official Roboflow split honored via filename tags
  (emv_train_* -> train, emv_valid_* -> val, emv_test_* -> test).
- MIO-TCD: official mirror split honored via tags
  (mitcd_train_/mitcd_val_/mitcd_test_).
- DAWN: stratified 70/15/15 by lowest-tier class (weather-balanced).
- Cap on images whose *only* class is `car` (MIO-TCD dominates).
- Writes data/dataset/{train,val,test}.txt + data.yaml + split manifest.

Cleanup note: partial output of the previous copy-based merge lives in
data/dataset_old (deleted here once, then recreated as txt-list dataset).
"""
import os
import sys
import glob
import random
import shutil
import collections
import time

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, PROC_DIR, DATASET_DIR, REPORTS_DIR  # noqa: E402

SRC_DIRS = {
    "miotcd": os.path.join(PROC_DIR, "miotcd_yolo"),
    "emergency": os.path.join(PROC_DIR, "emergency_yolo"),
    "dawn": os.path.join(PROC_DIR, "dawn_yolo"),
}
CLASS_NAMES = CONFIG["CLASS_NAMES"]
TIER_MAP = CONFIG["TIER_MAP"]
CAR_ONLY_CAP = 6000          # per-source cap on car-only images
SEED = 42


def load_classes(lbl_path):
    out = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                out.append(int(parts[0]))
    return out


def max_tier(classes):
    return min(TIER_MAP[CLASS_NAMES[c]] for c in classes)


def main():
    rng = random.Random(SEED)
    items = []
    for src_name, src in SRC_DIRS.items():
        img_dir, lbl_dir = os.path.join(src, "images"), os.path.join(src, "labels")
        if not os.path.isdir(img_dir):
            print(f"WARNING: {src_name} not prepared, skipping")
            continue
        n = 0
        last_report = time.monotonic()
        for ip in glob.glob(os.path.join(img_dir, "*.*")):
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(lp):
                continue
            classes = load_classes(lp)
            if not classes:
                continue
            base = os.path.basename(ip)
            weather = "clear"
            if base.startswith("dawn_"):
                weather = "dawn_" + base.split("_")[1]
            items.append((ip, lp, src_name, classes, weather))
            n += 1
            if time.monotonic() - last_report >= 30:
                print(f"{src_name}: scanned {n} labeled images", flush=True)
                last_report = time.monotonic()
        print(f"{src_name}: {n} images")

    # ---- cap car-only images per source -------------------------------
    per_src_car = collections.Counter()
    kept, dropped = [], 0
    for it in items:
        if set(it[3]) == {CLASS_NAMES.index("car")}:
            if per_src_car[it[2]] >= CAR_ONLY_CAP:
                dropped += 1
                continue
            per_src_car[it[2]] += 1
        kept.append(it)
    print(f"car-only cap ({CAR_ONLY_CAP}/src): kept {len(kept)}, dropped {dropped}")
    items = kept

    # ---- split ----------------------------------------------------------
    ratios = CONFIG["SPLIT_RATIOS"]
    fixed = {"train": [], "val": [], "test": []}
    flexible = []
    for it in items:
        tag = os.path.basename(it[0]).lower()
        if tag.startswith("emv_train_"):
            fixed["train"].append(it)
        elif tag.startswith("emv_valid_"):
            fixed["val"].append(it)
        elif tag.startswith("emv_test_"):
            fixed["test"].append(it)
        elif tag.startswith("mitcd_train_"):
            fixed["train"].append(it)
        elif tag.startswith("mitcd_val_"):
            fixed["val"].append(it)
        elif tag.startswith("mitcd_test_"):
            fixed["test"].append(it)
        else:
            flexible.append(it)

    by_tier = collections.defaultdict(list)
    for it in flexible:
        by_tier[max_tier(it[3])].append(it)
    splits = dict(fixed)
    for tier in sorted(by_tier):
        group = by_tier[tier]
        rng.shuffle(group)
        n = len(group)
        n_val, n_test = int(n * ratios["val"]), int(n * ratios["test"])
        splits["val"] += group[:n_val]
        splits["test"] += group[n_val:n_val + n_test]
        splits["train"] += group[n_val + n_test:]

    # ---- write txt lists + manifest -------------------------------------
    old = DATASET_DIR + "_old"
    if os.path.isdir(old):
        shutil.rmtree(old)
    if os.path.isdir(DATASET_DIR):
        os.rename(DATASET_DIR, old)
        shutil.rmtree(old, ignore_errors=True)   # may be slow; ignore if AV-blocked

    os.makedirs(DATASET_DIR, exist_ok=True)
    stats = {s: collections.Counter() for s in splits}
    weather_stats = {s: collections.Counter() for s in splits}
    manifest = ["split,source,weather,file,classes"]
    for split, its in splits.items():
        lines = []
        for ip, lp, src, classes, weather in its:
            lines.append(ip.replace("\\", "/"))
            for c in set(classes):
                stats[split][CLASS_NAMES[c]] += 1
            weather_stats[split][weather] += 1
            manifest.append(f"{split},{src},{weather},{os.path.basename(ip)},"
                            f"{'|'.join(CLASS_NAMES[c] for c in classes)}")
        with open(os.path.join(DATASET_DIR, f"{split}.txt"), "w") as f:
            f.write("\n".join(lines))
        print(f"{split}: {len(lines)} images -> {split}.txt")

    data_yaml = {
        "path": DATASET_DIR.replace("\\", "/"),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "split_manifest.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest))

    print("\n=== split summary (image counts per class) ===")
    for split in ("train", "val", "test"):
        print(f"{split}: {sum(weather_stats[split].values())} images")
        for c in CLASS_NAMES:
            print(f"   {c:14s} {stats[split][c]:7d}")
        print("   weather:", dict(weather_stats[split]))
    print(f"\ndataset -> {DATASET_DIR}")
    print(f"manifest -> {os.path.join(REPORTS_DIR, 'split_manifest.csv')}")


if __name__ == "__main__":
    main()
