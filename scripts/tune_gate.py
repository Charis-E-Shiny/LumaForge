"""Choose the fast/quality noise gate on validation scenes only.

Scores both deployed models on every degraded validation crop, then sweeps the gate. The chosen
gate sends the most captures to the fast model while the mean PSNR stays within `--tolerance`
of always using the quality model. Writes results/gate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import OnnxArm

from lumaforge.data import RAW_SCALE
from lumaforge.degrade import random_degradation
from lumaforge.isp.pipeline import ISPConfig, run_classical
from lumaforge.metrics import psnr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/crops"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/gate.json"))
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2, help="degradations drawn per crop")
    parser.add_argument("--tolerance", type=float, default=0.05, help="allowed PSNR loss in dB")
    args = parser.parse_args()

    entries = json.loads((args.data / "val.json").read_text())
    bayer_store = np.load(args.data / "bayer.npy", mmap_mode="r")
    rgb_store = np.load(args.data / "rgb.npy", mmap_mode="r")
    fast = OnnxArm(args.models / "lumanet_compact_int8.onnx", ISPConfig(denoise=False))
    quality = OnnxArm(args.models / "lumanet_learned_int8.onnx")

    rows = []
    for position, entry in enumerate(entries):
        bayer = np.asarray(bayer_store[entry["row"]], dtype=np.float32) / RAW_SCALE
        rgb = np.asarray(rgb_store[entry["row"]], dtype=np.float32) / 255.0
        top = ((bayer.shape[0] - args.crop) // 4) * 2
        left = ((bayer.shape[1] - args.crop) // 4) * 2
        bayer = bayer[top : top + args.crop, left : left + args.crop].copy()
        truth = rgb[top : top + args.crop, left : left + args.crop]
        for repeat in range(args.repeats):
            rng = np.random.default_rng(position * 10 + repeat)
            degraded, log = random_degradation(bayer, rng)
            compensation = 1.0 / log.get("exposure", 1.0)
            controls = run_classical(degraded, entry["camera"], compensation=compensation).controls
            rows.append((controls.denoise_sigma * controls.exposure,
                         psnr(fast(degraded, entry["camera"], compensation), truth),
                         psnr(quality(degraded, entry["camera"], compensation), truth)))

    table = np.array(rows)
    noise, fast_db, quality_db = table.T
    reference = float(quality_db.mean())
    sweep = []
    for gate in sorted({round(float(q), 4) for q in np.quantile(noise, np.linspace(0, 1, 41))} | {0.02}):
        uses_fast = noise < gate
        routed = float(np.where(uses_fast, fast_db, quality_db).mean())
        sweep.append({"gate": gate, "fast_share": float(uses_fast.mean()), "psnr": routed})
    allowed = [row for row in sweep if row["psnr"] >= reference - args.tolerance]
    chosen = max(allowed, key=lambda row: (row["fast_share"], -row["gate"]))
    output = {"samples": len(rows), "always_fast": float(fast_db.mean()), "always_quality": reference,
              "tolerance_db": args.tolerance, "chosen": chosen, "sweep": sweep}
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1))
    print(json.dumps({k: v for k, v in output.items() if k != "sweep"}, indent=1))
    for row in sweep[::5]:
        print(row)


if __name__ == "__main__":
    main()
