"""Kaggle download helper using KGAT bearer tokens (no username/password needed).

Tokens are read from the team-provided token file. The first token is used
for every request, as required by the project instructions.

Usage:
    python kaggle_client.py download <owner/slug> <dest_dir>
    python kaggle_client.py info <owner/slug>
"""
import os
import sys
import json
import time
import shutil
import requests

TOKEN_FILE = r"D:\Research\Group 3 Fall (Nishat Mustaqim)\K_GLM\EEG DEFENCE\kaggle.txt"
BASE = "https://www.kaggle.com/api/v1"


def load_tokens():
    tokens = []
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("KGAT_"):
                tokens.append(line)
    if not tokens:
        raise RuntimeError(f"No KGAT_ tokens found in {TOKEN_FILE}")
    return tokens


class KaggleClient:
    def __init__(self):
        self.tokens = load_tokens()
        self.session = requests.Session()
        self.token_idx = 0
        self.session.headers["Authorization"] = f"Bearer {self.tokens[0]}"

    def get(self, url, stream=False, **kw):
        return self.session.get(url, stream=stream, timeout=kw.pop("timeout", 60), **kw)

    def dataset_info(self, slug):
        r = self.get(f"{BASE}/datasets/view/{slug}")
        r.raise_for_status()
        return r.json()

    def dataset_download(self, slug, dest_dir):
        """Download (and extract) a dataset zip to dest_dir."""
        os.makedirs(dest_dir, exist_ok=True)
        url = f"{BASE}/datasets/download/{slug}"
        zip_path = os.path.join(dest_dir, slug.replace("/", "__") + ".zip")
        with self.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            t0 = time.time()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and time.time() - t0 > 10:
                        pct = 100 * done / total
                        print(f"\r  {pct:5.1f}%  {done/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
            print()
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"  downloaded {size_mb:.0f} MB, extracting ...")
        extract_dir = os.path.join(dest_dir, slug.split("/")[-1])
        shutil.unpack_archive(zip_path, extract_dir, format="zip")
        os.remove(zip_path)  # free space
        print(f"  extracted to {extract_dir}")
        return extract_dir


if __name__ == "__main__":
    cmd = sys.argv[1]
    kc = KaggleClient()
    if cmd == "info":
        info = kc.dataset_info(sys.argv[2])
        print(json.dumps({k: info.get(k) for k in ("title", "subtitle", "totalBytes", "usabilityRating", "licenseName")}, indent=2))
        files = info.get("files") or []
        print(f"{len(files)} files")
        for f in files[:30]:
            print(f"  {f.get('name'):70s} {f.get('totalBytes', 0)/1e6:9.1f} MB")
    elif cmd == "download":
        kc.dataset_download(sys.argv[2], sys.argv[3])
