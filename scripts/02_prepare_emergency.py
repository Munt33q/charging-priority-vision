"""Prepare the Kaggle emergency-vehicle dataset (prachiii33/dataset, the
Roboflow "Emergency Vehicle Detection" v3, CC BY 4.0).

It ships pre-split YOLO data (train/valid/test) with classes
[ambulance, car, fire_truck, police]. We remap ids to our unified class list
and PRESERVE the official split by tagging filenames: emv_<split>_<stem>.

Output: data/processed/emergency_yolo/{images,labels}
"""
import os
import sys
import glob
import shutil

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, RAW_DIR, PROC_DIR  # noqa: E402

SRC = os.path.join(RAW_DIR, "dataset")
OUT_DIR = os.path.join(PROC_DIR, "emergency_yolo")
SPLITS = ("train", "valid", "test")

# dataset class name -> unified class name
EMERGENCY_MAP = CONFIG["EMERGENCY_MAP"]


def main():
    import yaml
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    # read original class order
    with open(os.path.join(SRC, "data.yaml"), "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = [n.lower() for n in meta["names"]]
    print("dataset classes:", names)

    id_map = {}
    for i, n in enumerate(names):
        unified = EMERGENCY_MAP.get(n)
        id_map[i] = CONFIG["CLASS_NAMES"].index(unified) if unified else None
    print("id map (old->new):", id_map)

    counts = {s: 0 for s in SPLITS}
    for split in SPLITS:
        img_dir = os.path.join(SRC, split, "images")
        lbl_dir = os.path.join(SRC, split, "labels")
        if not os.path.isdir(img_dir):
            print(f"missing split dir: {img_dir}")
            continue
        for ip in glob.glob(os.path.join(img_dir, "*.*")):
            if not ip.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(os.path.basename(ip))[0]
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.isfile(lp):
                continue  # skip unlabeled positives
            out_lines = []
            with open(lp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    new_id = id_map.get(int(parts[0]))
                    if new_id is None:
                        continue
                    out_lines.append(f"{new_id} " + " ".join(parts[1:]))
            if not out_lines:
                continue
            base = f"emv_{split}_{stem}"
            shutil.copy(ip, os.path.join(OUT_DIR, "images", base + os.path.splitext(ip)[1].lower()))
            with open(os.path.join(OUT_DIR, "labels", base + ".txt"), "w") as f:
                f.write("\n".join(out_lines))
            counts[split] += 1

    print("converted:", counts, "| total", sum(counts.values()))


if __name__ == "__main__":
    main()
