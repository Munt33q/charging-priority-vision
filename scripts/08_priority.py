"""Priority assignment: model output -> 4-tier priority queue.

Stage 1 (vision): fine-grained vehicle detection (8 classes).
Stage 2 (this mapping): fine class -> priority tier, configurable in config.yaml.
  tier 0 EMERGENCY   (ambulance / fire_truck / police)
  tier 1 CAR
  tier 2 SEMI_TRUCK  (articulated)
  tier 3 TRUCK       (single unit)
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, MODELS_DIR  # noqa: E402

TIER_NAMES = {0: "EMERGENCY", 1: "CAR", 2: "SEMI_TRUCK", 3: "TRUCK"}


class PriorityAssigner:
    def __init__(self, weights=None, conf=None, imgsz=None):
        from ultralytics import YOLO
        default_weights = os.path.join(MODELS_DIR, "runs", "charging_priority", "weights", "best.pt")
        self.weights = weights or default_weights
        if not os.path.isfile(self.weights):
            raise FileNotFoundError(
                f"Model weights not found: {self.weights}. "
                "Download the completed Kaggle best.pt artifact first."
            )
        self.model = YOLO(self.weights)
        self.conf = CONFIG["CONF_THRESHOLD"] if conf is None else conf
        self.imgsz = CONFIG["IMG_SIZE"] if imgsz is None else imgsz

    def predict(self, frame_bgr):
        """Return list of dicts: class, tier, confidence, box(x1,y1,x2,y2)."""
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                                 verbose=False)[0]
        out = []
        for b in res.boxes:
            cls_id = int(b.cls.item())
            name = res.names[cls_id]
            out.append({
                "class": name,
                "tier": CONFIG["TIER_MAP"][name],
                "tier_name": TIER_NAMES[CONFIG["TIER_MAP"][name]],
                "confidence": float(b.conf.item()),
                "box": [round(float(v), 1) for v in b.xyxy[0].tolist()],
            })
        out.sort(key=lambda d: d["tier"])          # highest priority first
        return out

    def annotate(self, frame_bgr):
        """Draw tier-colored boxes + labels; returns annotated frame."""
        dets = self.predict(frame_bgr)
        colors = {0: (0, 0, 255), 1: (0, 200, 0), 2: (0, 140, 255), 3: (255, 0, 0)}
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            c = colors[d["tier"]]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
            label = f"T{d['tier']} {d['class']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
            cv2.putText(frame_bgr, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr, dets


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "0"   # image path or camera index
    pa = PriorityAssigner()
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ann, dets = pa.annotate(frame)
            cv2.imshow("charging priority", ann)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        frame = cv2.imread(src)
        if frame is None:
            raise SystemExit(f"Could not read image: {src}")
        ann, dets = pa.annotate(frame)
        out = os.path.splitext(src)[0] + "_tiers.jpg"
        cv2.imwrite(out, ann)
        for d in dets:
            print(d)
        print("saved:", out)
