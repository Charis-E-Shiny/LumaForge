"""Stream SIDD-CC train captures and cut them into a compact, CFA-normalised crop cache.

Each capture is roughly 75 MB, so originals are deleted as soon as crops are saved. The
job is resumable: instances with a finished ``.npz`` are skipped.

Usage:
    python scripts/prepare_sidd_cc.py --out data/cache --crops 24 --size 256 --workers 3
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
BASE = "https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/resolve/main"
API = "https://huggingface.co/api/datasets/MhdMohammadi/SIDD-CC/tree/main"
RAW_SCALE = 30000.0  # int16 quantisation, keeps negative sub-black noise


def fetch(url: str, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read()
        except Exception as exc:  # network errors are expected on long jobs
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} {url.split('/')[-1]}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def rggb_offset(cfa: str) -> tuple[int, int]:
    """Return the (row, col) shift that puts an R sample at the crop origin."""
    index = cfa.lower().index("r")
    return index // 2, index % 2


def process(instance: str, out: Path, crops: int, size: int) -> str:
    target = out / f"{instance}.npz"
    if target.exists():
        return f"skip {instance}"
    listing = json.loads(fetch(f"{API}/Raw/train/Data/{instance}"))
    names = sorted(e["path"].split("/")[-1] for e in listing)
    noisy_name = next(n for n in names if "_NOISY_RAW_" in n)
    meta_name = noisy_name.replace("NOISY", "METADATA")
    srgb_name = noisy_name.replace("NOISY_RAW", "GT_SRGB").replace(".MAT", ".PNG")

    noisy = scipy.io.loadmat(io.BytesIO(fetch(f"{BASE}/Raw/train/Data/{instance}/{noisy_name}")))["data"]
    meta = scipy.io.loadmat(io.BytesIO(fetch(f"{BASE}/Raw/train/Data/{instance}/{meta_name}")))
    png = fetch(f"{BASE}/SRGB/train/Data/{instance}/{srgb_name}")
    srgb = np.array(Image.open(io.BytesIO(png)).convert("RGB"))

    cfa = str(meta["cfa_str"][0])
    dr, dc = rggb_offset(cfa)
    height, width = noisy.shape
    # A deterministic seed per instance keeps the cache reproducible.
    rng = np.random.default_rng(abs(hash_name(instance)) % (2**32))
    rows = rng.integers(0, (height - size - 2) // 2, crops) * 2 + dr
    cols = rng.integers(0, (width - size - 2) // 2, crops) * 2 + dc
    bayer = np.stack([noisy[r:r + size, c:c + size] for r, c in zip(rows, cols)])
    rgb = np.stack([srgb[r:r + size, c:c + size] for r, c in zip(rows, cols)])
    bayer = np.clip(bayer * RAW_SCALE, -32000, 32000).astype(np.int16)

    parts = instance.split("_")
    info = {
        "instance": instance,
        "camera": parts[2],
        "iso": int(parts[3]),
        "shutter": int(parts[4]),
        "illuminant": int(parts[5]),
        "brightness": parts[6],
        "cfa": cfa,
        "as_shot_neutral": np.asarray(meta["AsShotNeutral"], dtype=float).ravel().tolist(),
        "color_matrix2": np.asarray(meta["ColorMatrix2"], dtype=float).ravel().tolist(),
    }
    temp = target.with_suffix(".tmp.npz")
    np.savez(temp, bayer=bayer, rgb=rgb, info=json.dumps(info))
    temp.replace(target)
    return f"done {instance} cfa={cfa} iso={parts[3]}"


def hash_name(name: str) -> int:
    value = 0
    for char in name:
        value = (value * 131 + ord(char)) % (2**61 - 1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cache"))
    parser.add_argument("--crops", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    text = fetch(f"{BASE}/Raw/train/Scene_Instances.txt").decode()
    instances = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"{len(instances)} instances", flush=True)

    def task(instance: str) -> None:
        try:
            print(process(instance, args.out, args.crops, args.size), flush=True)
        except Exception as exc:
            print(f"FAILED {instance}: {exc!r}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(task, instances))
    print("PREPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
