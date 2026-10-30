"""Evaluate the trained model on the test split, broken down per class and
per weather condition (clear / dawn_* tags in filenames).

Writes reports/eval_report.md with mAP50, precision, recall per class.
"""
import os
import sys
import json
import collections

sys.path.insert(0, os.path.dirname(__file__))
from paths import CONFIG, DATASET_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402


def main():
    from ultralytics import YOLO

    weights = os.path.join(MODELS_DIR, "runs", "charging_priority", "weights", "best.pt")
    if len(sys.argv) > 1:
        weights = sys.argv[1]
    model = YOLO(weights)

    data = os.path.join(DATASET_DIR, "data.yaml")
    metrics = model.val(data=data, split="test", imgsz=CONFIG["IMG_SIZE"],
                        conf=0.25, device=0, plots=True,
                        project=os.path.join(REPORTS_DIR, "eval"), name="test",
                        exist_ok=True)

    names = model.names
    lines = ["# Test-split evaluation report", "",
             f"weights: `{weights}`", "",
             "## Overall", ""]
    d = metrics.results_dict
    lines += [f"- mAP50-95: **{d.get('metrics/mAP50-95(B)', 0):.3f}**",
              f"- mAP50: **{d.get('metrics/mAP50(B)', 0):.3f}**",
              f"- precision: **{d.get('metrics/precision(B)', 0):.3f}**",
              f"- recall: **{d.get('metrics/recall(B)', 0):.3f}**", "",
              "## Per class", "",
              "| class | tier | images | instances | mAP50 | mAP50-95 |",
              "|---|---|---:|---:|---:|---:|"]
    tier_of = {n: CONFIG["TIER_MAP"][n] for n in CONFIG["CLASS_NAMES"]}
    per = {k: v for k, v in metrics.__dict__.items()}
    try:
        ap50 = metrics.box.ap50
        ap = metrics.box.ap
        cls_idx = metrics.box.ap_class_index
        n_img = getattr(metrics.box, "n_images", "")  # may not exist in all versions
        for row in cls_idx:
            i = int(row[0]) if hasattr(row, "__len__") else int(row)
            n = names[i]
            lines.append(f"| {n} | {tier_of.get(n,'-')} | {n_img} | - | "
                         f"{ap50[i]:.3f} | {ap[i].mean():.3f} |")
    except Exception as e:
        lines.append(f"(per-class table unavailable: {e})")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "eval_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nreport ->", os.path.join(REPORTS_DIR, "eval_report.md"))


if __name__ == "__main__":
    main()
