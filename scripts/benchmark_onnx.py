"""Benchmark ONNX model size, CPU inference latency and throughput."""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8.")

    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("Warmup must be non-negative and runs must be positive.")

    onnx.checker.check_model(str(args.onnx))

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )

    size = args.size

    inputs = {
        "packed": np.random.rand(1, 4, size, size).astype(np.float32),
        "noise": np.random.rand(1, 1, size, size).astype(np.float32),
        "base": np.random.rand(1, 3, size * 2, size * 2).astype(np.float32),
    }

    input_names = {item.name for item in session.get_inputs()}
    inputs = {key: value for key, value in inputs.items()
              if key in input_names}

    output_names = [item.name for item in session.get_outputs()]

    for _ in range(args.warmup):
        session.run(output_names, inputs)

    timings = []

    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(output_names, inputs)
        elapsed = time.perf_counter() - start
        timings.append(elapsed * 1000)

    average = float(np.mean(timings))
    median = float(np.median(timings))
    p95 = float(np.percentile(timings, 95))
    fps = 1000.0 / average

    model_size = args.onnx.stat().st_size / (1024 * 1024)

    print("\n===== ONNX PERFORMANCE RESULTS =====")
    print(f"Model:              {args.onnx.name}")
    print(f"Model size:         {model_size:.2f} MB")
    print(f"Execution provider: CPUExecutionProvider")
    print(f"Input resolution:   {size * 2} x {size * 2}")
    print(f"Warmup runs:        {args.warmup}")
    print(f"Benchmark runs:     {args.runs}")
    print(f"Average latency:    {average:.3f} ms")
    print(f"Median latency:     {median:.3f} ms")
    print(f"P95 latency:        {p95:.3f} ms")
    print(f"Throughput:         {fps:.2f} FPS")
    print("BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
