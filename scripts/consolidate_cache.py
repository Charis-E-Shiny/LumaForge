"""Merge per-instance crop files into memory-mappable arrays and write the splits.

Validation scenes are chosen per camera (every ninth instance) so that validation covers all
five sensors. The SIDD-CC test split is held out completely and is never touched here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("data/crops"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.cache.glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    first = np.load(files[0])
    crops, height, width = first["bayer"].shape
    total = crops * len(files)
    bayer = np.lib.format.open_memmap(args.out / "bayer.npy", "w+", np.int16, (total, height, width))
    rgb = np.lib.format.open_memmap(args.out / "rgb.npy", "w+", np.uint8, (total, height, width, 3))

    per_camera: dict[str, list[str]] = defaultdict(list)
    infos = {}
    for position, path in enumerate(files):
        data = np.load(path)
        info = json.loads(str(data["info"]))
        start = position * crops
        bayer[start : start + crops] = data["bayer"]
        rgb[start : start + crops] = data["rgb"]
        info["start"] = start
        infos[info["instance"]] = info
        per_camera[info["camera"]].append(info["instance"])

    validation = {name for names in per_camera.values() for name in sorted(names)[4::9]}
    splits: dict[str, list] = {"train": [], "val": []}
    for name, info in infos.items():
        split = "val" if name in validation else "train"
        for offset in range(crops):
            splits[split].append({"row": info["start"] + offset, "camera": info["camera"], "instance": name})
    for split, rows in splits.items():
        (args.out / f"{split}.json").write_text(json.dumps(rows))
    (args.out / "instances.json").write_text(json.dumps(infos, indent=1))
    bayer.flush()
    rgb.flush()
    print({k: len(v) for k, v in splits.items()}, "val scenes:", len(validation))


if __name__ == "__main__":
    main()
