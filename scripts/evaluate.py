"""Evaluate every arm on the held-out SIDD-CC test split.

Tasks:
    main       traditional, learned, hybrid and compressed models on all 928 clean blocks
    robust     all arms under each named degradation (every fourth block)
    ablation   classical stage removals and trained variants (every fourth block)

Writes one JSON per task to results/, including per-block rows for failure analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_model

from lumaforge.degrade import CONDITIONS, apply_condition
from lumaforge.hybrid import prepare
from lumaforge.isp import classical as c
from lumaforge.isp.pipeline import CALIBRATION, ISPConfig, run_classical
from lumaforge.metrics import LpipsMetric, delta_e, psnr, ssim

SHAPE_TO_CAMERA = {"3036x4048": "GP", "3120x4208": "N6", "2988x5312": "G4", "3024x4032": "IP", "3000x5328": "S6"}


class TestSet:
    def __init__(self, root: Path) -> None:
        self.noisy = scipy.io.loadmat(root / "raw_noisy.mat")["data"]
        self.rgb = scipy.io.loadmat(root / "srgb_gt.mat")["data"]
        lines = (root / "test_scenes_info.txt").read_text().splitlines()[1:]
        self.scenes = [(line.split("\t")[0], SHAPE_TO_CAMERA[line.split("\t")[2]]) for line in lines]

    def block(self, s: int, b: int) -> tuple[np.ndarray, np.ndarray, str]:
        camera = self.scenes[s][1]
        cfa = CALIBRATION["cfa"][camera]
        bayer = c.to_rggb(self.noisy[s, b].astype(np.float32), cfa)
        index = cfa.index("r")
        target = self.rgb[s, b, index // 2 :, index % 2 :].astype(np.float32) / 255.0
        return bayer, target[: bayer.shape[0], : bayer.shape[1]], camera


class TorchArm:
    def __init__(self, path: Path) -> None:
        self.model, checkpoint = load_model(path)
        self.isp = ISPConfig(**checkpoint.get("isp", {}))

    @torch.no_grad()
    def __call__(self, bayer: np.ndarray, camera: str, compensation: float) -> np.ndarray:
        inputs = prepare(bayer, camera, self.isp, compensation)
        tensors = [torch.from_numpy(v)[None] for v in (inputs.packed, inputs.noise, inputs.base)]
        return self.model(*tensors)[0].clamp(0, 1).numpy().transpose(1, 2, 0)


class OnnxArm:
    def __init__(self, path: Path, isp: ISPConfig | None = None) -> None:
        self.isp = isp or ISPConfig()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.session.get_inputs()}

    def __call__(self, bayer: np.ndarray, camera: str, compensation: float) -> np.ndarray:
        inputs = prepare(bayer, camera, self.isp, compensation)
        feed = {"packed": inputs.packed[None], "noise": inputs.noise[None], "base": inputs.base[None]}
        feed = {k: v for k, v in feed.items() if k in self.names}
        return np.clip(self.session.run(["rgb"], feed)[0][0].transpose(1, 2, 0), 0, 1)


class AdaptiveArm:
    """Routes each capture to the fast or the quality model; this is the deployed system."""

    def __init__(self, fast, quality) -> None:
        self.fast, self.quality = fast, quality
        self.routed = {"fast": 0, "quality": 0}

    def __call__(self, bayer: np.ndarray, camera: str, compensation: float) -> np.ndarray:
        route = run_classical(bayer, camera, compensation=compensation).controls.route
        self.routed[route] += 1
        return (self.fast if route == "fast" else self.quality)(bayer, camera, compensation)


def classical_arm(config: ISPConfig):
    def run(bayer: np.ndarray, camera: str, compensation: float) -> np.ndarray:
        return run_classical(bayer, camera, config, compensation=compensation).rgb
    return run


def score(arms: dict, data: TestSet, blocks: list[tuple[int, int]], lpips: LpipsMetric,
          condition: dict | None = None, keep_rows: bool = False) -> dict:
    """Score arms on the same degraded inputs, so that differences are paired."""
    condition = condition or {}
    compensation = 1.0 / condition.get("exposure", 1.0)
    totals = {name: {"psnr": [], "ssim": [], "delta_e": [], "lpips": []} for name in arms}
    rows = []
    for s, b in blocks:
        bayer, target, camera = data.block(s, b)
        # A fresh generator per block gives every arm identical noise.
        rng = np.random.default_rng(s * 1000 + b)
        degraded = apply_condition(bayer, condition, rng)
        degraded = degraded[: degraded.shape[0] - degraded.shape[0] % 16,
                            : degraded.shape[1] - degraded.shape[1] % 16]
        truth = target[: degraded.shape[0], : degraded.shape[1]]
        row = {"scene": data.scenes[s][0], "camera": camera, "block": b, "scene_index": s}
        for name, arm in arms.items():
            output = arm(degraded, camera, compensation)
            values = {"psnr": psnr(output, truth), "ssim": ssim(output, truth),
                      "delta_e": delta_e(output, truth), "lpips": lpips(output, truth)}
            for key, value in values.items():
                totals[name][key].append(value)
            row[name] = values
        rows.append(row)
    summary = {name: {k: float(np.mean(v)) for k, v in table.items()} for name, table in totals.items()}
    return {"summary": summary, "rows": rows if keep_rows else None, "blocks": len(blocks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["main", "robust", "ablation"])
    parser.add_argument("--data", type=Path, default=Path("data/sidd_cc/test"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, default=0, help="debug: only this many blocks")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)
    torch.set_num_threads(4)

    data = TestSet(args.data)
    lpips = LpipsMetric()
    every = [(s, b) for s in range(len(data.scenes)) for b in range(32)]
    subset = every[::4]
    if args.limit:
        every, subset = every[: args.limit], subset[: args.limit]
    started = time.time()

    traditional = classical_arm(ISPConfig())
    fixed = classical_arm(ISPConfig(adaptive=False))
    teacher = TorchArm(args.checkpoints / "teacher.pt")
    student = TorchArm(args.checkpoints / "student.pt")
    learned = TorchArm(args.checkpoints / "learned.pt")
    compact = TorchArm(args.checkpoints / "abl_no_classical_denoise.pt")

    if args.task == "main":
        compact_int8 = OnnxArm(args.models / "lumanet_compact_int8.onnx", ISPConfig(denoise=False))
        learned_int8 = OnnxArm(args.models / "lumanet_learned_int8.onnx")
        adaptive = AdaptiveArm(compact_int8, learned_int8)
        arms = {
            "traditional_fixed": fixed,
            "traditional": traditional,
            "learned_fp32": learned,
            "learned_int8": learned_int8,
            "hybrid_compact_fp32": compact,
            "hybrid_compact_int8": compact_int8,
            "hybrid_quality_fp32": teacher,
            "hybrid_quality_fp16": OnnxArm(args.models / "lumanet_quality_fp16.onnx"),
            "hybrid_fast_fp32": student,
            "hybrid_fast_int8": OnnxArm(args.models / "lumanet_fast_int8.onnx"),
            "hybrid_adaptive": adaptive,
        }
        output = score(arms, data, every, lpips, keep_rows=True)
        output["routed"] = adaptive.routed
        (args.out / "main.json").write_text(json.dumps(output))

    elif args.task == "robust":
        table = {}
        for name, condition in CONDITIONS.items():
            arms = {"traditional": traditional, "learned": learned,
                    "hybrid_adaptive": AdaptiveArm(compact, learned)}
            table[name] = score(arms, data, subset, lpips, condition)
            table[name]["condition"] = condition
            table[name]["routed"] = arms["hybrid_adaptive"].routed
            print(name, json.dumps(table[name]["summary"]), flush=True)
            (args.out / "robust.json").write_text(json.dumps(table, indent=1))

        # Failure mode: the capture is 2 stops under, but the gain metadata is missing.
        blind = {"exposure": 0.25}
        arms = {"traditional": traditional, "hybrid_adaptive": AdaptiveArm(compact, learned)}
        blind_arms = {name: (lambda f: lambda b, cam, _: f(b, cam, 1.0))(arm) for name, arm in arms.items()}
        table["blind_underexposed_2stop"] = score(blind_arms, data, subset, lpips, blind)
        table["blind_underexposed_2stop"]["condition"] = {"exposure": 0.25, "metadata": "missing"}
        (args.out / "robust.json").write_text(json.dumps(table, indent=1))

    else:
        base = ISPConfig()
        classical = {
            "full": base,
            "no_denoise": replace(base, denoise=False),
            "no_white_balance": replace(base, white_balance=False),
            "no_ccm": replace(base, ccm=False),
            "no_tone": replace(base, tone=False),
            "no_chroma_filter": replace(base, chroma=False),
            "no_adaptive": replace(base, adaptive=False),
        }
        trained = {}
        for name in ("abl_base", "abl_no_noise_map", "abl_no_color_head", "abl_no_degrade",
                     "abl_no_color_loss", "abl_no_classical_denoise", "abl_learned"):
            path = args.checkpoints / f"{name}.pt"
            if path.exists():
                trained[name] = TorchArm(path)
        classical_arms = {name: classical_arm(cfg) for name, cfg in classical.items()}
        severe = CONDITIONS["combined_severe"]
        output = {
            "classical": score(classical_arms, data, subset, lpips),
            "trained_clean": score(trained, data, subset, lpips),
            "trained_severe": score(trained, data, subset, lpips, severe),
        }
        (args.out / "ablation.json").write_text(json.dumps(output, indent=1))
    print(f"EVAL_DONE {args.task} in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
