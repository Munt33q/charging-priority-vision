"""Package existing splits and labels, without changing local training data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import yaml

from kaggle_client import load_tokens

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kaggle_metadata_upload"
REF = "mluyxp11/charging-priority-metadata"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {}
    labels = {}
    for split in ("train", "val", "test"):
        entries = []
        for line in (ROOT / "data" / "dataset" / f"{split}.txt").read_text().splitlines():
            image = Path(line.strip())
            if not line.strip() or "/aug/images/" in line.replace("\\", "/"):
                continue
            relative = image.relative_to(ROOT / "data" / "processed").as_posix()
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            label_relative = str(Path(relative).parent.parent / "labels" / label.name).replace("\\", "/")
            entries.append(relative)
            labels[label_relative] = label
        splits[split] = entries
        print(f"{split}: {len(entries)} original images", flush=True)
    if any(not entries for entries in splits.values()):
        raise RuntimeError("An existing split is empty")
    sets = [set(entries) for entries in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Split path overlap detected")
    names = yaml.safe_load((ROOT / "data" / "dataset" / "data.yaml").read_text())["names"]
    # Non-ZIP suffix prevents Kaggle from expanding the compact archive on upload.
    with zipfile.ZipFile(OUT / "metadata.bundle", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("splits.json", json.dumps({"splits": splits, "names": names}))
        for index, (relative, label) in enumerate(labels.items(), 1):
            archive.write(label, "labels/" + relative)
            if index % 10000 == 0:
                print(f"Packed {index}/{len(labels)} labels", flush=True)
    with zipfile.ZipFile(OUT / "metadata.bundle") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(labels) + 1
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "Charging Priority Metadata", "id": REF,
        "licenses": [{"name": "other"}],
        "description": "Split manifests and labels derived from the private charging-priority-data bundle. Original source licenses and restrictions apply; not relicensed as CC0."
    }))
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = load_tokens()[0]
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create",
                    "-p", str(OUT)], env=env, check=True)
    print("Uploaded private metadata dataset:", REF, flush=True)


if __name__ == "__main__":
    main()
