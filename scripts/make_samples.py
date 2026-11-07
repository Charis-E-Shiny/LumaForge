"""Build the demo captures by stitching contiguous SIDD-CC test blocks.

Test blocks sit on a regular grid, so neighbouring blocks form a larger seamless crop. Only
held-out test scenes are used, which means the demo shows images the networks never saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SHAPE_TO_CAMERA

from lumaforge.isp import classical as c
from lumaforge.isp.analysis import analyze
from lumaforge.isp.pipeline import CALIBRATION

RAW_SCALE = 30000.0
CAMERA_NAMES = {"GP": "Google Pixel", "N6": "Nexus 6", "G4": "LG G4", "IP": "iPhone 7", "S6": "Galaxy S6"}


def largest_grid(blocks: np.ndarray) -> list[list[int]]:
    """Return a rectangular grid of block indices, in reading order."""
    rows = sorted(set(blocks[:, 0].tolist()))
    cols = sorted(set(blocks[:, 1].tolist()))
    lookup = {(int(r), int(k)): i for i, (r, k, _, _) in enumerate(blocks)}
    best: list[list[int]] = []
    for top in range(len(rows)):
        for bottom in range(top, len(rows)):
            for left in range(len(cols)):
                for right in range(left, len(cols)):
                    grid = [[lookup.get((r, k), -1) for k in cols[left : right + 1]] for r in rows[top : bottom + 1]]
                    continuous = all(b - a == 256 for a, b in zip(rows[top:bottom], rows[top + 1 : bottom + 1]))
                    continuous &= all(b - a == 256 for a, b in zip(cols[left:right], cols[left + 1 : right + 1]))
                    if continuous and min(min(g) for g in grid) >= 0 and sum(map(len, grid)) > sum(map(len, best)):
                        best = grid
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sidd_cc/test"))
    parser.add_argument("--out", type=Path, default=Path("samples"))
    parser.add_argument("--thumbs", type=Path, default=Path("public/samples"))
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    args.thumbs.mkdir(parents=True, exist_ok=True)

    noisy = scipy.io.loadmat(args.data / "raw_noisy.mat")["data"]
    rgb = scipy.io.loadmat(args.data / "srgb_gt.mat")["data"]
    blocks = scipy.io.loadmat(args.data / "BenchmarkBlocks32.mat")["BenchmarkBlocks32"].astype(int)
    lines = (args.data / "test_scenes_info.txt").read_text().splitlines()[1:]
    grid = largest_grid(blocks)
    print("grid", len(grid), "x", len(grid[0]))
    # Keep the response small enough for the function's 4.5 MB limit.
    grid = [row[:4] for row in grid[:2]]

    scored = []
    for s, line in enumerate(lines):
        camera = SHAPE_TO_CAMERA[line.split("\t")[2]]
        stats = analyze(c.pack(noisy[s, grid[0][0]].astype(np.float32)))
        brightness = float(rgb[s, grid[0][0]].mean())
        detail = float(np.mean([rgb[s, b].std() for row in grid for b in row]))
        scored.append((s, camera, stats.sigma, brightness, detail, line.split("\t")[0]))

    # One capture per camera first, the most detailed, then fill by noise diversity.
    chosen, seen = [], set()
    for row in sorted(scored, key=lambda r: -r[4]):
        if row[1] not in seen:
            seen.add(row[1])
            chosen.append(row)
    for row in sorted(scored, key=lambda r: -r[2]):
        if len(chosen) < args.count and row not in chosen:
            chosen.append(row)

    index = []
    for s, camera, sigma, _, _, scene in chosen[: args.count]:
        bayer = np.vstack([np.hstack([noisy[s, b] for b in row]) for row in grid]).astype(np.float32)
        image = np.vstack([np.hstack([rgb[s, b] for b in row]) for row in grid])
        cfa = CALIBRATION["cfa"][camera]
        mosaic = c.to_rggb(bayer, cfa)
        offset = cfa.index("r")
        target = image[offset // 2 :, offset % 2 :][: mosaic.shape[0], : mosaic.shape[1]]
        sample_id = f"scene-{scene}"
        info = {"camera": camera, "scene": scene, "sigma": sigma}
        np.savez_compressed(args.out / f"{sample_id}.npz", bayer=np.clip(mosaic * RAW_SCALE, -32000, 32000).astype(np.int16),
                            rgb=target, info=json.dumps(info))
        Image.fromarray(target[:256, :256]).resize((160, 160)).save(args.thumbs / f"{sample_id}.jpg", quality=86)
        index.append({"id": sample_id, "title": f"Scene {scene}", "camera": CAMERA_NAMES[camera],
                      "iso": "unknown", "thumbnail": f"/samples/{sample_id}.jpg"})
        print(index[-1], f"sigma={sigma:.4f}", mosaic.shape)
    (args.out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
