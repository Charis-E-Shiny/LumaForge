"""Compare a trained PyTorch checkpoint with its exported ONNX model."""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8.")

    onnx.checker.check_model(str(args.onnx))

    torch.manual_seed(42)
    model, _ = load_model(args.checkpoint)
    model.eval()

    size = args.size

    packed = torch.rand(1, 4, size, size)
    noise = torch.rand(1, 1, size, size)
    base = torch.rand(1, 3, size * 2, size * 2)

    with torch.no_grad():
        expected = model(packed, noise, base).cpu().numpy()

    session = ort.InferenceSession(
        str(args.onnx),
        providers=["CPUExecutionProvider"],
    )

    inputs = {
        "packed": packed.numpy(),
        "noise": noise.numpy(),
        "base": base.numpy(),
    }

    actual = session.run(["rgb"], inputs)[0]

    if expected.shape != actual.shape:
        raise AssertionError(
            f"Shape mismatch: PyTorch {expected.shape}, ONNX {actual.shape}"
        )

    difference = np.abs(expected - actual)
    max_error = float(difference.max())
    mean_error = float(difference.mean())

    print(f"PyTorch output shape: {expected.shape}")
    print(f"ONNX output shape:   {actual.shape}")
    print(f"Maximum difference:  {max_error:.8f}")
    print(f"Mean difference:     {mean_error:.8f}")

    np.testing.assert_allclose(
        expected,
        actual,
        rtol=args.rtol,
        atol=args.atol,
        err_msg="PyTorch and ONNX outputs differ beyond the allowed tolerance.",
    )

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
