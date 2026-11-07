"""Sanity checks on the SIDD-CC test blocks: value ranges, CFA phase and scene statistics.

For every candidate CFA phase, renders a block and scores it against the sRGB target. The
winning phase per camera confirms the phase table used by evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumaforge.isp import classical as c
from lumaforge.isp.analysis import analyze
from lumaforge.isp.pipeline import run_classical

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sidd_cc/test")
noisy = scipy.io.loadmat(root / "raw_noisy.mat")["data"]
gt_rgb = scipy.io.loadmat(root / "srgb_gt.mat")["data"]
blocks = scipy.io.loadmat(root / "BenchmarkBlocks32.mat")["BenchmarkBlocks32"]
lines = (root / "test_scenes_info.txt").read_text().splitlines()[1:]
shape_camera = {"3036x4048": "GP", "3120x4208": "N6", "2988x5312": "G4", "3024x4032": "IP", "3000x5328": "S6"}
scenes = [(ln.split("\t")[0], shape_camera[ln.split("\t")[2]]) for ln in lines]
print("noisy", noisy.shape, noisy.dtype, float(noisy.min()), float(noisy.max()))
print("rgb", gt_rgb.shape, gt_rgb.dtype)
print("block rows odd:", int((blocks[:, 0] - 1).sum() % 2), "first blocks", blocks[:4].tolist())
print("row parity set", set(((blocks[:, 0] - 1) % 2).tolist()), "col parity set", set(((blocks[:, 1] - 1) % 2).tolist()))

patterns = ["rggb", "grbg", "gbrg", "bggr"]
tally: dict[str, dict[str, list[float]]] = {}
for s, (scene, camera) in enumerate(scenes):
    for b in range(0, 32, 8):
        for pattern in patterns:
            bayer = c.to_rggb(noisy[s, b], pattern).astype(np.float32)
            result = run_classical(bayer, camera)
            index = pattern.index("r")
            target = gt_rgb[s, b, index // 2 :, index % 2 :][: bayer.shape[0], : bayer.shape[1]] / 255.0
            mse = float(np.mean((result.rgb - target) ** 2))
            tally.setdefault(camera, {}).setdefault(pattern, []).append(10 * np.log10(1 / mse))
for camera, table in tally.items():
    print(camera, {k: round(float(np.mean(v)), 2) for k, v in table.items()})

stats = [analyze(c.pack(noisy[s, b].astype(np.float32))) for s in range(29) for b in (3, 19)]
print("sigma pct", np.percentile([x.sigma for x in stats], [5, 50, 95]).round(5))
print("p99 pct", np.percentile([x.p99 for x in stats], [5, 25, 50, 75, 95]).round(4))
print("median pct", np.percentile([x.median for x in stats], [5, 50, 95]).round(4))
