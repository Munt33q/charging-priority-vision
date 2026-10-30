"""Kaggle kernel: train charging-priority YOLO on the uploaded dataset.

Runs on NvidiaTeslaP100/T4. Inputs:
  /kaggle/input/charging-priority-data/charging-priority-data.zip
Outputs (saved to kernel output):
  best.pt, last.pt, results.csv, eval summary printed to log
"""
import os
import shutil
import subprocess
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import zipfile

WORK = "/kaggle/working"
PROJ = os.path.join(WORK, "charging-priority-data")

# Flush diagnostics immediately, including failures before training starts.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
print("Starting Kaggle GPU training", flush=True)
STARTED = time.monotonic()


def progress(stage, **details):
    record = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": round(time.monotonic() - STARTED), **details}
    Path(WORK, "progress.json").write_text(json.dumps(record, indent=2))
    print("PROGRESS:", json.dumps(record), flush=True)


progress("checking_gpu", eta="Pending measured epoch speed; queue time is unknown")
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU assigned. Enable a Kaggle GPU accelerator before retrying.")
gpu_info = {"torch": torch.__version__, "devices": [
    {"name": torch.cuda.get_device_name(i),
     "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
    for i in range(torch.cuda.device_count())]}
print("GPU:", json.dumps(gpu_info), flush=True)
Path(WORK, "gpu_info.json").write_text(json.dumps(gpu_info, indent=2))

# Install the training dependency explicitly; retain Kaggle's CUDA torch build.
progress("installing_dependencies")
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--retries", "2", "--timeout", "60",
                "ultralytics==8.4.192"], check=True, timeout=900)
from ultralytics import YOLO

# Kaggle can expand uploaded ZIPs and can use owner-qualified input mounts.
inputs = Path("/kaggle/input")
progress("preparing_inputs")
print("Input mounts:", [str(p) for p in inputs.iterdir()])
# Never recursively walk the image dataset just to locate its root.
mounts = [inputs / "charging-priority-data",
          inputs / "datasets" / "mluyxp11" / "charging-priority-data"]
candidates = [root for mount in mounts for root in
              (mount, mount / "charging-priority-data")]
archives = [mount / "charging-priority-data.zip" for mount in mounts
            if (mount / "charging-priority-data.zip").is_file()]
if archives:
    print("Extracting", archives[0])
    with zipfile.ZipFile(archives[0]) as z:
        z.extractall(WORK)
else:
    roots = [root for root in candidates
             if (root / "config.yaml").is_file()
             and (root / "scripts" / "04_merge_split.py").is_file()
             and (root / "data" / "processed").is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one project bundle in Kaggle inputs, found {roots}")
    source = roots[0]
    print("Using input images in place from", source, flush=True)
    Path(PROJ, "scripts").mkdir(parents=True, exist_ok=True)
    Path(PROJ, "data").mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "yolov8s.pt"):
        shutil.copy2(source / name, Path(PROJ, name))
    for script in (source / "scripts").glob("*.py"):
        shutil.copy2(script, Path(PROJ, "scripts", script.name))
    # Keep the project writable without copying tens of thousands of images.
    processed_link = Path(PROJ, "data", "processed")
    if not processed_link.exists():
        processed_link.symlink_to(source / "data" / "processed", target_is_directory=True)
if not Path(PROJ, "config.yaml").is_file():
    raise RuntimeError(f"Project config missing after input preparation: {PROJ}")

os.chdir(PROJ)
sys_path = os.path.join(PROJ, "scripts")
for name, content in globals().get("BUNDLED_SCRIPTS", {}).items():
    Path(sys_path, name).write_text(content, encoding="utf-8")
progress("inputs_ready", mode="Original images referenced in place; no full dataset copy")

# 2. Restore precomputed splits and local labels. No remote label scanning.
progress("restoring_precomputed_splits")
import yaml
metadata_mounts = [inputs / "charging-priority-metadata",
                   inputs / "datasets" / "mluyxp11" / "charging-priority-metadata"]
metadata_dir = Path(WORK, "prepared_metadata")
metadata_dir.mkdir(exist_ok=True)
for mount in metadata_mounts:
    if (mount / "metadata.bundle").is_file():
        with zipfile.ZipFile(mount / "metadata.bundle") as archive:
            archive.extractall(metadata_dir)
        break
    if (mount / "splits.json").is_file():
        metadata_dir = mount
        break
else:
    raise RuntimeError("Missing charging-priority-metadata input")
metadata = json.loads((metadata_dir / "splits.json").read_text())
dataset_dir = Path(PROJ, "data", "dataset")
dataset_dir.mkdir(parents=True, exist_ok=True)
local_processed = Path(WORK, "training_sources")
source_processed = Path(PROJ, "data", "processed")
count = 0
for split, entries in metadata["splits"].items():
    paths = []
    for relative in entries:
        image = local_processed / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        if not image.is_symlink():
            image.symlink_to(source_processed / relative)
        label_relative = Path(relative).parent.parent / "labels" / (Path(relative).stem + ".txt")
        label = local_processed / label_relative
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata_dir / "labels" / label_relative, label)
        paths.append(str(image))
        count += 1
        if count % 5000 == 0:
            progress("restoring_precomputed_splits", prepared_images=count)
    (dataset_dir / f"{split}.txt").write_text("\n".join(paths) + "\n")
    print(f"Prepared {split}: {len(paths)} images", flush=True)
with (dataset_dir / "data.yaml").open("w") as handle:
    yaml.safe_dump({"path": str(dataset_dir), "train": "train.txt", "val": "val.txt",
                    "test": "test.txt", "names": {int(k): v for k, v in metadata["names"].items()}}, handle)
progress("splits_ready", original_images=count)

# 3. CPU image preprocessing; only model training/evaluation use the GPU.
progress("weather_augmentation")
subprocess.run([sys.executable, "-u", os.path.join(sys_path, "05_augment_weather.py")],
               check=True, timeout=5400)

# 4. train
import torch
from ultralytics import YOLO

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "")
model = YOLO(os.path.join(PROJ, "yolov8s.pt"))
epoch_times = []
epoch_started = None


def start_epoch(trainer):
    global epoch_started
    epoch_started = time.monotonic()


def report_eta(trainer):
    # This event is after validation/checkpointing, unlike on_train_epoch_end.
    if epoch_started is None:
        return
    epoch_times.append(time.monotonic() - epoch_started)
    seconds_per_epoch = sum(epoch_times[-3:]) / len(epoch_times[-3:])
    completed = trainer.epoch + 1
    remaining = max(0, trainer.epochs - completed)
    # Allow one extra epoch-equivalent for final validation and test evaluation.
    seconds_left = seconds_per_epoch * (remaining + 1)
    progress("training", completed_epochs=completed, target_epochs=trainer.epochs,
             seconds_per_epoch=round(seconds_per_epoch, 1),
             estimated_remaining_minutes=round(seconds_left / 60, 1),
             estimated_completion_utc=(datetime.now(timezone.utc) +
                                       timedelta(seconds=seconds_left)).isoformat(),
             eta_basis="Last up to 3 full epochs, plus evaluation allowance; early stopping may finish sooner",
             eta_provisional=len(epoch_times) < 3,
             gpu=torch.cuda.get_device_name(0))


model.add_callback("on_train_epoch_start", start_epoch)
model.add_callback("on_fit_epoch_end", report_eta)
progress("starting_training", eta="First estimate after epoch 1 including validation")
model.train(
    data=os.path.join(PROJ, "data", "dataset", "data.yaml"),
    epochs=40,
    imgsz=640,
    batch=16,          # Conservative batch for either Kaggle T4 or P100.
    patience=12,
    save_period=1,
    project=os.path.join(WORK, "runs"),
    name="charging_priority",
    exist_ok=True,
    hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
    degrees=0, translate=0.1, scale=0.4, shear=0,
    perspective=0.0, flipud=0, fliplr=0.3,
    mosaic=1.0, mixup=0.1, copy_paste=0.0,
    workers=4,
    seed=42,
    device=0,
)

# 5. copy weights to kernel output
run_dir = os.path.join(WORK, "runs", "charging_priority")
for f in ("best.pt", "last.pt", "results.csv"):
    src = os.path.join(run_dir, "weights", f) if f.endswith(".pt") else os.path.join(run_dir, f)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(WORK, f))
        print("saved output:", f)

# 6. quick test-split evaluation
best = os.path.join(WORK, "best.pt")
if os.path.isfile(best):
    progress("test_evaluation")
    m = YOLO(best)
    metrics = m.val(data=os.path.join(PROJ, "data", "dataset", "data.yaml"),
                    split="test", imgsz=640, device=0)
    print("TEST mAP50-95:", metrics.results_dict.get("metrics/mAP50-95(B)"))
    print("TEST mAP50:", metrics.results_dict.get("metrics/mAP50(B)"))
progress("complete")
print("DONE")
