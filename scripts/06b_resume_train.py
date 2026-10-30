"""Resume local training from last.pt (used after an interruption)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from paths import MODELS_DIR


def main():
    from ultralytics import YOLO
    last = os.path.join(MODELS_DIR, "runs", "charging_priority", "weights", "last.pt")
    model = YOLO(last)
    model.train(resume=True)
    print("resumed training finished")


if __name__ == "__main__":
    main()
