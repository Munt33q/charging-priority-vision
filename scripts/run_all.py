"""Run the full pipeline end-to-end:
  1. prepare MIO-TCD (from the official tar OR the Kaggle YOLO mirror)
  2. prepare emergency dataset
  3. prepare DAWN
  4. merge + split
  5. weather augmentation
  6. train
  7. evaluate
Usage: python run_all.py [skip_to_step]
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("01 MIO-TCD prepare", "01_prepare_miotcd.py"),
    ("02 emergency prepare", "02_prepare_emergency.py"),
    ("03 DAWN prepare", "03_prepare_dawn.py"),
    ("04 merge + split", "04_merge_split.py"),
    ("05 weather augment", "05_augment_weather.py"),
    ("06 train", "06_train.py"),
    ("07 evaluate", "07_evaluate.py"),
]


def run(script):
    print(f"\n{'=' * 60}\n>>> {script}\n{'=' * 60}")
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    if r.returncode != 0:
        raise SystemExit(f"step failed: {script}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i, (name, script) in enumerate(STEPS, 1):
        if i < start:
            continue
        print(f"\n### step {i}: {name}")
        run(script)
    print("\nALL DONE")
