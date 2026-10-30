"""Prepare the Kaggle upload bundle + dataset metadata, then upload via kaggle CLI.

Bundle layout (mirrors the local project so paths.py works unchanged):
  charging-priority-data/
    config.yaml
    yolov8s.pt
    data/processed/{miotcd_yolo, emergency_yolo, dawn_yolo}
    scripts/{paths.py, 04_merge_split.py, 05_augment_weather.py,
             weather_aug.py, 06_train.py}

Then zips it to data/kaggle_upload.zip and creates the private Kaggle
dataset mluyxp11/charging-priority-data.
"""
import os
import sys
import json
import shutil
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
STAGE = os.path.join(ROOT, "data", "kaggle_stage")
ZIP_PATH = os.path.join(ROOT, "data", "kaggle_upload.zip")
USER = "mluyxp11"
SLUG = "charging-priority-data"
DATASET_REF = f"{USER}/{SLUG}"

SCRIPTS = ["paths.py", "04_merge_split.py", "05_augment_weather.py",
           "weather_aug.py", "06_train.py", "kaggle_train_kernel.py"]


def build_stage():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE, ignore_errors=True)
    for sub in ("miotcd_yolo", "emergency_yolo", "dawn_yolo"):
        src = os.path.join(PROC, sub)
        dst = os.path.join(STAGE, "data", "processed", sub)
        print(f"copying {sub} ...", flush=True)
        # robocopy is ~50x faster than shutil.copytree for many small files
        r = subprocess.run(["robocopy", src, dst, "/E", "/MT:16", "/NFL", "/NDL",
                            "/NJH", "/NP"], capture_output=True, text=True)
        if r.returncode > 7:
            raise RuntimeError(f"robocopy failed for {sub}: rc={r.returncode}")
    os.makedirs(os.path.join(STAGE, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "config.yaml"), STAGE)
    shutil.copy(os.path.join(ROOT, "yolov8s.pt"), STAGE)
    for s in SCRIPTS:
        shutil.copy(os.path.join(ROOT, "scripts", s), os.path.join(STAGE, "scripts", s))
    print("stage built at", STAGE, flush=True)


def make_zip():
    if os.path.isfile(ZIP_PATH):
        os.remove(ZIP_PATH)
    print("zipping (store mode - jpgs are already compressed) ...")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED, compresslevel=None) as z:
        for root, _, files in os.walk(STAGE):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, STAGE)
                z.write(full, os.path.join("charging-priority-data", rel))
    print(f"zip: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1e9:.2f} GB)")


def kaggle_env():
    """Set KAGGLE_API_TOKEN for the CLI (new CLI token style)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from kaggle_client import load_tokens
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = load_tokens()[0]
    return env


def upload():
    env = kaggle_env()
    meta = {
        "title": "Charging Priority Data",
        "id": DATASET_REF,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    up_dir = os.path.join(ROOT, "data", "kaggle_upload_dir")
    if os.path.isdir(up_dir):
        shutil.rmtree(up_dir)
    os.makedirs(up_dir)
    shutil.copy(ZIP_PATH, os.path.join(up_dir, "charging-priority-data.zip"))
    with open(os.path.join(up_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)
    print("uploading dataset (this streams ~2.4 GB) ...", flush=True)
    kaggle_cmd = [sys.executable, "-m", "kaggle"]
    r = subprocess.run(kaggle_cmd + ["datasets", "create", "-p", up_dir],
                       env=env, capture_output=True, text=True)
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    if r.returncode != 0:
        # maybe exists -> new version
        print("create failed; trying new version ...", flush=True)
        r = subprocess.run(kaggle_cmd + ["datasets", "version", "-p", up_dir,
                                         "-m", "update"],
                           env=env, capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:])


if __name__ == "__main__":
    if "--upload-only" in sys.argv and os.path.isfile(ZIP_PATH):
        print("zip exists, skipping stage build")
    else:
        build_stage()
        make_zip()
    upload()
