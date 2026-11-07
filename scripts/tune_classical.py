"""Choose the white-balance prior weight and sharpening strength for the baseline.

Uses cached training crops only. The white-balance weight minimises angular error against
the camera's as-shot neutral. Sharpening is reported as a PSNR/edge trade-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumaforge.isp import classical as c
from lumaforge.isp.analysis import analyze, plan
from lumaforge.isp.pipeline import CALIBRATION, CALIBRATION_PATH, run_classical
from lumaforge.metrics_lite import delta_e_76, psnr

RAW_SCALE = 30000.0


def angle(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    priors = CALIBRATION["wb_prior"]
    weights = (0.0, 0.25, 0.5, 0.65, 0.8, 0.9, 1.0)
    errors = {w: [] for w in weights}
    scores: dict[str, list] = {"with_asshot": [], "estimated": [], "delta_as": [], "delta_est": []}
    sigma_rows = []

    for path in sorted(args.cache.glob("*.npz"))[::3]:
        if path.name.endswith(".tmp.npz"):
            continue
        data = np.load(path)
        info = json.loads(str(data["info"]))
        neutral = np.asarray(info["as_shot_neutral"], dtype=np.float32)
        truth = neutral[1] / neutral
        prior = np.asarray(priors[info["camera"]], dtype=np.float32)
        for bayer, rgb in zip(data["bayer"][:4], data["rgb"][:4]):
            mosaic = bayer.astype(np.float32) / RAW_SCALE
            packed = c.pack(mosaic)
            stats = analyze(packed)
            denoised = c.denoise_packed(packed, stats.sigma)
            for weight in weights:
                estimate = c.estimate_white_balance(denoised, prior=prior, prior_weight=weight)
                # Compare illuminants, the inverse of gains.
                errors[weight].append(angle(1 / estimate, 1 / truth))
            target = rgb.astype(np.float32) / 255.0
            asshot = run_classical(mosaic, info["camera"], gains=truth).rgb
            estimated = run_classical(mosaic, info["camera"]).rgb
            scores["with_asshot"].append(psnr(asshot, target))
            scores["estimated"].append(psnr(estimated, target))
            scores["delta_as"].append(delta_e_76(asshot, target))
            scores["delta_est"].append(delta_e_76(estimated, target))
            sigma_rows.append((info["iso"], stats.sigma, plan(stats).sharpen))

    table = {w: (round(float(np.mean(v)), 3), round(float(np.median(v)), 3)) for w, v in errors.items()}
    print("WB weight -> (mean, median) angular error, degrees", table)
    print({k: round(float(np.mean(v)), 3) for k, v in scores.items()})
    rows = np.asarray(sigma_rows)
    print("sigma percentiles", np.percentile(rows[:, 1], [10, 25, 50, 75, 90]).round(4))
    print("corr(sqrt iso, sigma)", np.corrcoef(np.sqrt(rows[:, 0]), rows[:, 1])[0, 1].round(3))
    best = min(table, key=lambda w: table[w][0])
    print("best weight", best)
    if args.write:
        payload = json.loads(CALIBRATION_PATH.read_text())
        payload["wb_weight"] = best
        CALIBRATION_PATH.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
