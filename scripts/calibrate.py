"""Fit the classical ISP to the training scenes: per-camera CCM and a global tone curve.

The baseline must be a fair one, so it is calibrated on the same data as the network. Only
training instances are used. As-shot white balance is used while fitting because it removes
illuminant error from the colour fit, but the pipeline itself still estimates white balance
blindly at run time, since the test blocks ship without metadata.

Writes lumaforge/isp/calibration.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lumaforge.isp import classical as c

RAW_SCALE = 30000.0


def load_instances(cache: Path):
    for path in sorted(cache.glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        data = np.load(path)
        yield json.loads(str(data["info"])), data["bayer"], data["rgb"]


def means(bayer: np.ndarray, rgb: np.ndarray, gains: np.ndarray, block: int = 8):
    """Average into blocks: white-balanced camera RGB against sRGB target, noise-free."""
    packed = c.pack(bayer.astype(np.float32) / RAW_SCALE)
    balanced = c.apply_white_balance(packed, gains)
    camera = np.stack((balanced[..., 0], balanced[..., 1:3].mean(-1), balanced[..., 3]), -1)
    size = (camera.shape[0] // block) * block
    camera = camera[:size, :size].reshape(size // block, block, size // block, block, 3).mean((1, 3))
    target = rgb[: size * 2, : size * 2].astype(np.float32) / 255.0
    target = target.reshape(size // block, block * 2, size // block, block * 2, 3).mean((1, 3))
    return camera.reshape(-1, 3), target.reshape(-1, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("lumaforge/isp/calibration.json"))
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    pairs: dict[str, list] = defaultdict(list)
    dng: dict[str, list] = defaultdict(list)
    cfa: dict[str, set] = defaultdict(set)
    gains_seen: dict[str, list] = defaultdict(list)
    # Validation scenes are excluded so that validation stays an honest estimate.
    val_path = args.cache.parent / "crops" / "val.json"
    held_out = {r["instance"] for r in json.loads(val_path.read_text())} if val_path.exists() else set()
    for info, bayers, rgbs in load_instances(args.cache):
        if info["instance"] in held_out:
            continue
        neutral = np.asarray(info["as_shot_neutral"], dtype=np.float32)
        gains = (neutral[1] / neutral).astype(np.float32)
        gains_seen[info["camera"]].append(gains)
        for bayer, rgb in zip(bayers[:8], rgbs[:8]):
            camera, target = means(bayer, rgb, gains)
            pairs[info["camera"]].append((camera, target))
        dng[info["camera"]].append(np.asarray(info["color_matrix2"]).reshape(3, 3))
        cfa[info["camera"]].add(info["cfa"])

    ccm_table = {}
    tone_x, tone_y = [], []
    for camera, rows in pairs.items():
        cam = np.concatenate([r[0] for r in rows])
        tgt = np.concatenate([r[1] for r in rows])
        keep = (tgt.max(1) < 0.97) & (tgt.min(1) > 0.02)
        linear_target = c.srgb_degamma(tgt[keep])
        dng_ccm = c.camera_to_srgb_matrix(np.mean(dng[camera], 0))
        fitted, *_ = np.linalg.lstsq(cam[keep], linear_target, rcond=None)
        fitted = fitted.T.astype(np.float32)
        ccm_table[camera] = fitted.tolist()
        predicted = cam[keep] @ fitted.T
        tone_x.append(predicted @ np.array([0.2126, 0.7152, 0.0722]))
        tone_y.append(linear_target @ np.array([0.2126, 0.7152, 0.0722]))
        if args.diagnose:
            ratio = np.median(linear_target / np.maximum(predicted, 1e-4), axis=0)
            print(camera, "cfa", cfa[camera], "n", int(keep.sum()), "median ratio target/pred", ratio.round(3))
            direct, *_ = np.linalg.lstsq(cam[keep], linear_target, rcond=None)
            print("  lstsq CCM", direct.T.round(3).tolist(), "\n  dng CCM", dng_ccm.round(3).tolist())

    x, y = np.concatenate(tone_x), np.concatenate(tone_y)
    valid = (x > 1e-4) & (y > 1e-4)
    x, y = x[valid], y[valid]
    if args.diagnose:
        for low, high in zip(np.quantile(x, np.linspace(0, 0.95, 8)), np.quantile(x, np.linspace(0.05, 1, 8))):
            inside = (x >= low) & (x < high)
            print(f"linear {x[inside].mean():.4f} -> target linear {np.median(y[inside]):.4f}")

    best = (1e9, None)
    for exposure in np.geomspace(0.5, 8.0, 28):
        for shoulder in (0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
            for toe in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
                scaled = x * exposure
                out = np.power(scaled * (1 + scaled / shoulder**2) / (1 + scaled), toe)
                error = float(np.mean((c.srgb_gamma(out) - c.srgb_gamma(y)) ** 2))
                if error < best[0]:
                    best = (error, {"exposure": float(exposure), "shoulder": shoulder, "toe": toe})
    print("tone fit", best, "gamma-space PSNR", -10 * np.log10(best[0]))

    ccm_table["generic"] = np.mean(list(ccm_table.values()), axis=0).tolist()
    payload = {
        "tone": best[1],
        "ccm": ccm_table,
        "cfa": {k: sorted(v)[0] for k, v in cfa.items()},
        "wb_prior": {k: np.median(v, 0).tolist() for k, v in gains_seen.items()},
    }
    payload["wb_prior"]["generic"] = np.median(np.concatenate(list(gains_seen.values())), 0).tolist()
    args.out.write_text(json.dumps(payload, indent=1))
    print("wrote", args.out, payload["cfa"], "wb prior", payload["wb_prior"])


if __name__ == "__main__":
    main()
