"""Compare trained checkpoints on validation scenes only, using all four quality metrics.

Validation PSNR alone cannot separate the candidates, because blind white-balance error dominates
it. Each candidate is ranked on PSNR, SSIM, CIEDE2000 and LPIPS; the lowest mean rank wins.
Every crop is scored clean and degraded. Writes results/selection.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import TorchArm, classical_arm

from lumaforge.data import RAW_SCALE
from lumaforge.degrade import random_degradation
from lumaforge.isp.pipeline import ISPConfig
from lumaforge.metrics import LpipsMetric, delta_e, psnr, ssim

CANDIDATES = ("student", "teacher", "learned", "abl_base", "abl_no_noise_map", "abl_no_color_head",
              "abl_no_degrade", "abl_no_color_loss", "abl_no_classical_denoise", "abl_learned")
LOWER_IS_BETTER = {"delta_e", "lpips"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/crops"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("results/selection.json"))
    parser.add_argument("--crop", type=int, default=256)
    args = parser.parse_args()

    entries = json.loads((args.data / "val.json").read_text())
    bayer_store = np.load(args.data / "bayer.npy", mmap_mode="r")
    rgb_store = np.load(args.data / "rgb.npy", mmap_mode="r")
    arms = {"traditional": classical_arm(ISPConfig())}
    for name in CANDIDATES:
        path = args.checkpoints / f"{name}.pt"
        if path.exists():
            arms[name] = TorchArm(path)
    lpips = LpipsMetric()
    totals = {name: {"psnr": [], "ssim": [], "delta_e": [], "lpips": []} for name in arms}

    for position, entry in enumerate(entries):
        bayer = np.asarray(bayer_store[entry["row"]], dtype=np.float32) / RAW_SCALE
        rgb = np.asarray(rgb_store[entry["row"]], dtype=np.float32) / 255.0
        top = ((bayer.shape[0] - args.crop) // 4) * 2
        left = ((bayer.shape[1] - args.crop) // 4) * 2
        bayer = bayer[top : top + args.crop, left : left + args.crop].copy()
        truth = rgb[top : top + args.crop, left : left + args.crop]
        degraded, log = random_degradation(bayer, np.random.default_rng(position))
        for mosaic, compensation in ((bayer, 1.0), (degraded, 1.0 / log.get("exposure", 1.0))):
            for name, arm in arms.items():
                output = arm(mosaic, entry["camera"], compensation)
                totals[name]["psnr"].append(psnr(output, truth))
                totals[name]["ssim"].append(ssim(output, truth))
                totals[name]["delta_e"].append(delta_e(output, truth))
                totals[name]["lpips"].append(lpips(output, truth))

    summary = {name: {k: float(np.mean(v)) for k, v in table.items()} for name, table in totals.items()}
    names = [name for name in summary if name != "traditional"]
    ranks = dict.fromkeys(names, 0.0)
    for metric in ("psnr", "ssim", "delta_e", "lpips"):
        order = sorted(names, key=lambda n: summary[n][metric], reverse=metric not in LOWER_IS_BETTER)
        for rank, name in enumerate(order, 1):
            ranks[name] += rank / 4
    output = {"crops": len(entries), "scored": 2 * len(entries), "summary": summary,
              "mean_rank": ranks, "selected": min(names, key=ranks.get)}
    args.out.write_text(json.dumps(output, indent=1))
    for name in sorted(names, key=ranks.get) + ["traditional"]:
        row = {k: round(v, 3) for k, v in summary[name].items()}
        print(f"{name:26s} rank {ranks.get(name, 0):5.2f} {row}", flush=True)
    print("SELECTED", output["selected"], flush=True)


if __name__ == "__main__":
    main()
