"""Measure parameters, MACs, file size, latency and peak memory for each ONNX model.

Each measurement runs in a fresh process, so peak memory is not polluted by earlier runs.
Numbers come from this machine's CPU, which the output records alongside the results.

    python scripts/benchmark.py            # measure everything, write results/benchmark.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {
    "hybrid_quality_fp32": ("lumanet_quality.onnx", "teacher"),
    "hybrid_quality_fp16": ("lumanet_quality_fp16.onnx", "teacher"),
    "hybrid_fast_fp32": ("lumanet_fast.onnx", "student"),
    "hybrid_fast_int8": ("lumanet_fast_int8.onnx", "student"),
    "hybrid_compact_fp32": ("lumanet_compact.onnx", "abl_no_classical_denoise"),
    "hybrid_compact_int8": ("lumanet_compact_int8.onnx", "abl_no_classical_denoise"),
    "learned_fp32": ("lumanet_learned.onnx", "learned"),
    "learned_int8": ("lumanet_learned_int8.onnx", "learned"),
}


def worker(path: str, side: int, threads: int) -> None:
    import numpy as np
    import onnxruntime as ort
    import psutil

    process = psutil.Process()
    baseline = process.memory_info().rss
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}
    feed = {
        "packed": np.random.rand(1, 4, side // 2, side // 2).astype(np.float32),
        "noise": np.random.rand(1, 1, side // 2, side // 2).astype(np.float32),
        "base": np.random.rand(1, 3, side, side).astype(np.float32),
    }
    feed = {k: v for k, v in feed.items() if k in names}
    session.run(["rgb"], feed)  # the first run allocates; it is not timed
    timings = []
    for _ in range(8):
        started = time.perf_counter()
        session.run(["rgb"], feed)
        timings.append((time.perf_counter() - started) * 1000)
    info = process.memory_info()
    peak = getattr(info, "peak_wset", info.rss)
    print(json.dumps({"ms_median": float(np.median(timings)), "ms_p95": float(np.percentile(timings, 95)),
                      "peak_mb": (peak - baseline) / 1e6}))


def macs_and_params(checkpoint: Path, side: int = 512) -> tuple[int, float]:
    import torch
    from train import load_model

    model, _ = load_model(checkpoint)
    counted = [0]

    def hook(module, _inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            counted[0] += kernel * module.in_channels // module.groups * output.numel()
        elif isinstance(module, torch.nn.Linear):
            counted[0] += module.in_features * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(torch.zeros(1, 4, side // 2, side // 2), torch.zeros(1, 1, side // 2, side // 2),
              torch.zeros(1, 3, side, side))
    for handle in handles:
        handle.remove()
    return sum(p.numel() for p in model.parameters()), counted[0] / 1e9


def classical_latency(side: int) -> float:
    import numpy as np

    from lumaforge.isp.pipeline import run_classical

    rng = np.random.default_rng(0)
    bayer = np.clip(rng.normal(0.2, 0.05, (side, side)), 0, 1).astype(np.float32)
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        run_classical(bayer, "IP")
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "benchmark.json")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
        return

    report: dict = {
        "machine": {"cpu": platform.processor(), "python": platform.python_version(),
                    "system": f"{platform.system()} {platform.release()}"},
        "classical_ms": {str(side): classical_latency(side) for side in (256, 512, 1024)},
        "models": {},
    }
    for name, (file_name, checkpoint) in MODELS.items():
        path = args.models / file_name
        params, gmacs = macs_and_params(ROOT / "checkpoints" / f"{checkpoint}.pt")
        row = {"file": file_name, "params": params, "gmacs_512": gmacs,
               "size_mb": path.stat().st_size / 1e6, "latency": {}}
        for side, threads in ((256, 1), (512, 1), (512, 4), (1024, 4)):
            output = subprocess.run(
                [sys.executable, __file__, "--worker", str(path), str(side), str(threads)],
                capture_output=True, text=True, check=True,
            ).stdout
            row["latency"][f"{side}px_{threads}t"] = json.loads(output.strip().splitlines()[-1])
        report["models"][name] = row
        print(name, json.dumps(row), flush=True)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
