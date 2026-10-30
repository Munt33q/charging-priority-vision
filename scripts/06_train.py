"""Train YOLO on the merged charging-station priority dataset.

Usage: python 06_train.py [model] [epochs]
Reads config.yaml for defaults; uses CUDA if available.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, DATASET_DIR, MODELS_DIR, ROOT  # noqa: E402


def main():
    import torch
    from ultralytics import YOLO

    print("CUDA:", torch.cuda.is_available(),
              "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")

    model_name = sys.argv[1] if len(sys.argv) > 1 else CONFIG["MODEL"]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else CONFIG["EPOCHS"]

    model = YOLO(model_name)
    results = model.train(
        data=os.path.join(DATASET_DIR, "data.yaml"),
        epochs=epochs,
        imgsz=CONFIG["IMG_SIZE"],
        batch=CONFIG["BATCH"],
        patience=CONFIG["PATIENCE"],
        project=os.path.join(MODELS_DIR, "runs"),
        name="charging_priority",
        exist_ok=True,
        # ultralytics' own geometric/color augmentation stays on; our weather
        # augmentation is already baked into the train images.
        hsv_h=0.01, hsv_s=0.5, hsv_v=0.4,
        degrees=0, translate=0.1, scale=0.4, shear=0,
        perspective=0.0, flipud=0, fliplr=0.3,
        mosaic=1.0, mixup=0.1, copy_paste=0.0,
        workers=4,
        seed=42,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    print("training done")
    print(results)

    # export best weights
    best = os.path.join(MODELS_DIR, "runs", "charging_priority", "weights", "best.pt")
    if os.path.isfile(best):
        print("best weights:", best)


if __name__ == "__main__":
    main()
