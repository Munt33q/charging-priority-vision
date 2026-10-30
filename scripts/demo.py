"""Run a reproducible charging-priority demo on an image or video.

Examples:
  python scripts/demo.py --source path/to/image.jpg
  python scripts/demo.py --source path/to/video.mp4 --output demo.mp4
  python scripts/demo.py --source 0

The demo performs detection and tier assignment only. It does not claim that a
vehicle is actively responding to an emergency, and it does not implement
tracking or charger-zone occupancy scheduling.
"""
import argparse
import json
import os
import sys
import importlib.util
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODELS_DIR  # noqa: E402

_priority_path = Path(__file__).with_name("08_priority.py")
_spec = importlib.util.spec_from_file_location("priority_impl", _priority_path)
_priority = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_priority)
PriorityAssigner = _priority.PriorityAssigner


def _default_weights():
    return os.path.join(MODELS_DIR, "runs", "charging_priority", "weights", "best.pt")


def _print_detections(detections):
    print(json.dumps(detections, indent=2))
    if detections:
        print(f"Highest priority detected: tier {detections[0]['tier']} ({detections[0]['tier_name']})")
    else:
        print("No vehicles detected above the confidence threshold.")


def image_demo(assigner, source, output):
    frame = cv2.imread(source)
    if frame is None:
        raise SystemExit(f"Could not read image: {source}")
    annotated, detections = assigner.annotate(frame)
    if not cv2.imwrite(output, annotated):
        raise SystemExit(f"Could not write demo output: {output}")
    _print_detections(detections)
    print(f"Saved annotated demo: {output}")


def video_demo(assigner, source, output, display):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video/camera: {source}")
    writer = None
    if output:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not create video output: {output}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated, detections = assigner.annotate(frame)
        count += 1
        if writer:
            writer.write(annotated)
        if display:
            cv2.imshow("charging priority demo", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    capture.release()
    if writer:
        writer.release()
        print(f"Saved {count} annotated frames: {output}")
    if display:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument("--weights", default=_default_weights())
    parser.add_argument("--output", help="Output image/video path")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    assigner = PriorityAssigner(args.weights, args.conf, args.imgsz)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if Path(args.source).suffix.lower() in image_exts:
        output = args.output or str(Path(args.source).with_name(Path(args.source).stem + "_priority.jpg"))
        image_demo(assigner, args.source, output)
    else:
        video_demo(assigner, args.source, args.output, args.display)


if __name__ == "__main__":
    main()
