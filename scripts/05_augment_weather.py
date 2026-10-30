"""Apply adverse-weather augmentations to TRAIN images only.

For a fraction of clean train images, generates augmented copies with the
same YOLO labels (boxes unchanged - weather effects don't move objects).
Validation/test stay untouched (real conditions only).

Augmented files go to data/dataset/aug/{images,labels} (ultralytics maps
.../images/... -> .../labels/... automatically) and are appended to train.txt.
"""
import os
import sys
import glob
import random
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from paths import DATASET_DIR  # noqa: E402
from weather_aug import augment_image  # noqa: E402

SEED = 123
AUG_FRACTION = 0.35


def main():
    rng = random.Random(SEED)
    aug_img = os.path.join(DATASET_DIR, "aug", "images")
    aug_lbl = os.path.join(DATASET_DIR, "aug", "labels")
    os.makedirs(aug_img, exist_ok=True)
    os.makedirs(aug_lbl, exist_ok=True)

    with open(os.path.join(DATASET_DIR, "train.txt")) as f:
        train_lines = [l.strip() for l in f if l.strip()]
    existing_aug = {os.path.basename(l) for l in train_lines
                    if "/aug/images/" in l.replace("\\", "/")}
    clear = [l for l in train_lines
             if "/aug/images/" not in l.replace("\\", "/")]
    rng.shuffle(clear)
    targets = clear[:int(len(clear) * AUG_FRACTION)]
    print(f"train: {len(train_lines)} ({len(existing_aug)} already augmented), "
          f"augmenting {len(targets)}")

    new_lines = []
    n = 0
    started = time.monotonic()
    last_report = started
    for ip in targets:
        img = cv2.imread(ip)
        if img is None:
            continue
        stem = os.path.splitext(os.path.basename(ip))[0]
        # find matching label (ultralytics convention: images->labels)
        lp = ip.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        lp = os.path.splitext(lp)[0] + ".txt"
        if not os.path.isfile(lp):
            continue
        out = augment_image(img, rng)
        out_path = os.path.join(aug_img, f"aug_{stem}.jpg")
        cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open(lp) as f:
            content = f.read()
        with open(os.path.join(aug_lbl, f"aug_{stem}.txt"), "w") as f:
            f.write(content)
        new_lines.append(out_path.replace("\\", "/"))
        n += 1
        now = time.monotonic()
        if n % 250 == 0 or now - last_report >= 30:
            remaining = (now - started) / n * (len(targets) - n)
            print(f"  weather: {n}/{len(targets)}, elapsed {now-started:.0f}s, "
                  f"estimated remaining {remaining/60:.1f} min", flush=True)
            last_report = now

    with open(os.path.join(DATASET_DIR, "train.txt"), "a") as f:
        f.write("\n" + "\n".join(new_lines))
    print(f"wrote {n} augmented train images; train.txt now has "
          f"{len(train_lines) + len(new_lines)} entries")


if __name__ == "__main__":
    main()
