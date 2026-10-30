"""Prepare the DAWN adverse-weather dataset (orvile/dawn-detection-in-adverse-weather-nature)
in YOLO format, using the PASCAL VOC xml annotations as ground truth.

Layout: DAWN/{Fog,Rain,Snow,Sand}/{W}/{W}_PASCAL_VOC/*.xml + images beside them.

Every output image is tagged with its weather condition in the filename so
the split script can keep weather balanced across splits.

Output: data/processed/dawn_yolo/{images,labels}
"""
import os
import sys
import glob
import shutil
import xml.etree.ElementTree as ET

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, RAW_DIR, PROC_DIR  # noqa: E402

OUT_DIR = os.path.join(PROC_DIR, "dawn_yolo")

# DAWN class names (lowercase) -> unified class; None = excluded
DAWN_MAP = {
    "car": "car",
    "pickup truck": "pickup_truck",
    "van": "work_van",
    "bus": None,
    "truck": None,          # ambiguous: could be semi or single-unit
    "motorcycle": None,
    "bicycle": None,
    "person": None,
    "ambulance": "ambulance",
    "fire truck": "fire_truck",
    "police": "police",
    "police car": "police",
}

WEATHERS = ["fog", "rain", "snow", "sand"]


def main():
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

    src_root = os.path.join(RAW_DIR, "dawn-detection-in-adverse-weather-nature", "DAWN")
    if not os.path.isdir(src_root):
        cands = glob.glob(os.path.join(RAW_DIR, "*dawn*", "**", "DAWN"), recursive=True)
        if not cands:
            raise RuntimeError("DAWN folder not found")
        src_root = cands[0]
    print("source:", src_root)

    counts = {"boxes": {}, "imgs": {}}
    n_out = 0
    for weather in WEATHERS:
        xmls = glob.glob(os.path.join(src_root, "**", f"{weather.capitalize()}_PASCAL_VOC", "*.xml"),
                         recursive=True)
        # fallback: any xml under a folder named like the weather
        if not xmls:
            xmls = [p for p in glob.glob(os.path.join(src_root, "**", "*.xml"), recursive=True)
                    if weather in os.path.basename(p).lower()
                    or weather in os.path.dirname(p).lower()]
        print(f"{weather}: {len(xmls)} xml files")
        n_w = 0
        for xp in xmls:
            try:
                root = ET.parse(xp).getroot()
            except ET.ParseError:
                continue
            fn = root.findtext("filename")
            if not fn:
                continue
            # image sits next to the xml or in a sibling folder
            ip = os.path.join(os.path.dirname(xp), fn)
            if not os.path.isfile(ip):
                hits = [p for p in glob.glob(os.path.join(src_root, "**", fn), recursive=True)
                        if p.lower().endswith((".jpg", ".jpeg", ".png"))]
                ip = hits[0] if hits else None
            if not ip:
                continue
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for obj in root.iter("object"):
                cls = (obj.findtext("name") or "").strip().lower()
                unified = DAWN_MAP.get(cls)
                if unified is None:
                    continue
                bb = obj.find("bndbox")
                x1 = max(0.0, float(bb.findtext("xmin", 0)))
                y1 = max(0.0, float(bb.findtext("ymin", 0)))
                x2 = min(float(w), float(bb.findtext("xmax", w)))
                y2 = min(float(h), float(bb.findtext("ymax", h)))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{CONFIG['CLASS_NAMES'].index(unified)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts["boxes"][unified] = counts["boxes"].get(unified, 0) + 1
            if not lines:
                continue
            stem = os.path.splitext(fn)[0]
            base = f"dawn_{weather}_{stem}"
            cv2.imwrite(os.path.join(OUT_DIR, "images", base + ".jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(OUT_DIR, "labels", base + ".txt"), "w") as f:
                f.write("\n".join(lines))
            n_out += 1
            n_w += 1
        counts["imgs"][weather] = n_w

    print(f"\nwrote {n_out} images -> {OUT_DIR}")
    print("per weather:", counts["imgs"])
    print("boxes per class:", counts["boxes"])


if __name__ == "__main__":
    main()
